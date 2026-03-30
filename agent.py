from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import pickle
import os

# ── Load resources ──
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)

with open("bm25_chunks.pkl", "rb") as f:
    data = pickle.load(f)
bm25 = BM25Okapi([t.lower().split() for t in data["texts"]])
chunk_texts = data["texts"]
chunk_metas = data["metas"]

api_key = None
try:
    import streamlit as st
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = "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"

llm = ChatGroq(
    api_key=api_key,
    model_name="llama-3.3-70b-versatile"
)

# ── Global uploaded vector DB (set from app.py) ──
uploaded_vector_db = None
uploaded_doc_name = None

def set_uploaded_db(db, name):
    global uploaded_vector_db, uploaded_doc_name
    uploaded_vector_db = db
    uploaded_doc_name = name

# ── Hybrid search helper ──
def agent_search(query, k=5):
    vector_results = vector_db.similarity_search(query, k=k)
    tokens = query.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    top_idx = sorted(range(len(bm25_scores)),
                     key=lambda i: bm25_scores[i], reverse=True)[:3]
    results = list(vector_results)
    for idx in top_idx:
        results.append(Document(
            page_content=chunk_texts[idx],
            metadata=chunk_metas[idx]
        ))
    return results[:6]

# ══════════════════════════════════════
# TOOLS
# ══════════════════════════════════════

def tool_search(query):
    """Search existing 10 company HR documents"""
    results = agent_search(query)
    context = "\n\n".join([r.page_content for r in results])
    sources = list(set([
        os.path.basename(r.metadata.get("source", "Unknown"))
        for r in results
    ]))
    return context, sources

def tool_search_uploaded(query):
    """Search the user-uploaded PDF document"""
    global uploaded_vector_db, uploaded_doc_name
    if uploaded_vector_db is None:
        return "No document has been uploaded yet.", []
    results = uploaded_vector_db.similarity_search(query, k=4)
    context = "\n\n".join([r.page_content for r in results])
    return context, [uploaded_doc_name or "Uploaded Document"]

def tool_search_both(query):
    """Search both existing company docs AND uploaded document"""
    context1, sources1 = tool_search(query)
    context2, sources2 = tool_search_uploaded(query)
    combined_context = f"FROM COMPANY DATABASE:\n{context1}\n\nFROM UPLOADED DOCUMENT:\n{context2}"
    return combined_context, sources1 + sources2

def tool_compare(company_a, company_b, topic):
    """Compare two companies on a topic"""
    def get_info(company):
        # Check if company matches uploaded doc
        if (uploaded_doc_name and
                company.lower() in uploaded_doc_name.lower()):
            ctx, _ = tool_search_uploaded(f"{topic} at {company}")
            return ctx
        results = vector_db.similarity_search(
            f"{topic} policy at {company}", k=3
        )
        return "\n".join([r.page_content for r in results])
    return get_info(company_a), get_info(company_b)

def tool_report(company_name):
    """Generate full HR policy report for a company"""
    topics = [
        "leave vacation sick casual",
        "compensation salary bonus",
        "parental maternity paternity",
        "work from home hybrid",
        "performance review appraisal",
        "learning development training"
    ]
    sections = {}

    # Check if it matches uploaded doc
    use_uploaded = (uploaded_vector_db is not None and
                    uploaded_doc_name and
                    company_name.lower() in uploaded_doc_name.lower())

    for topic in topics:
        if use_uploaded:
            ctx, _ = tool_search_uploaded(f"{topic}")
            sections[topic] = ctx[:400]
        else:
            results = vector_db.similarity_search(
                f"{topic} at {company_name}", k=2
            )
            if results:
                sections[topic] = "\n".join(
                    [r.page_content[:300] for r in results]
                )
    return sections

def tool_self_correct(question, answer):
    """Self-correct the answer"""
    prompt = f"""Review this answer and improve it if needed.
Question: {question}
Answer: {answer}

If the answer is good, return it as is.
If it needs improvement, return an improved version.
Return ONLY the final answer, nothing else."""
    response = llm.invoke(prompt)
    return response.content

# ══════════════════════════════════════
# MAIN AGENT FUNCTION
# ══════════════════════════════════════

def run_agent(question: str, chat_history: list = [],
              use_uploaded: bool = False) -> dict:
    tools_used = []
    steps = 0

    try:
        # ── Step 1: Classify question ──
        steps += 1
        uploaded_context = ""
        if uploaded_vector_db is not None:
            uploaded_context = f"There is also an uploaded document available: {uploaded_doc_name}."

        classify_prompt = f"""Classify this HR policy question into one of these types:
1. SEARCH - general policy question about existing companies
2. COMPARE - comparing two companies
3. REPORT - requesting full company summary
4. UPLOADED - question specifically about the uploaded document
5. BOTH - question that needs both existing docs and uploaded doc

{uploaded_context}

Question: {question}
Reply with just the type: SEARCH, COMPARE, REPORT, UPLOADED, or BOTH"""

        classification = llm.invoke(classify_prompt).content.strip().upper()
        for qtype in ["SEARCH", "COMPARE", "REPORT", "UPLOADED", "BOTH"]:
            if qtype in classification:
                classification = qtype
                break
        else:
            classification = "UPLOADED" if use_uploaded else "SEARCH"

        # ── Step 2: Use the right tool ──
        context = ""
        sources = []

        if classification == "UPLOADED" or use_uploaded:
            steps += 1
            tools_used.append("search_uploaded_document")
            context, sources = tool_search_uploaded(question)

        elif classification == "BOTH":
            steps += 1
            tools_used.append("search_uploaded_document")
            tools_used.append("search_hr_documents")
            context, sources = tool_search_both(question)

        elif classification == "COMPARE":
            steps += 1
            tools_used.append("compare_companies")

            extract_prompt = f"""From this question extract:
1. Company A name
2. Company B name
3. Topic being compared

Question: {question}
Reply in this exact format:
COMPANY_A: name
COMPANY_B: name
TOPIC: topic"""

            extracted = llm.invoke(extract_prompt).content
            lines = extracted.strip().split("\n")
            company_a = "Google"
            company_b = "TCS"
            topic = "leave"
            for line in lines:
                if "COMPANY_A:" in line:
                    company_a = line.split(":", 1)[1].strip()
                elif "COMPANY_B:" in line:
                    company_b = line.split(":", 1)[1].strip()
                elif "TOPIC:" in line:
                    topic = line.split(":", 1)[1].strip()

            info_a, info_b = tool_compare(company_a, company_b, topic)
            context = (f"INFO ABOUT {company_a}:\n{info_a}\n\n"
                      f"INFO ABOUT {company_b}:\n{info_b}")

        elif classification == "REPORT":
            steps += 1
            tools_used.append("generate_company_report")

            extract_prompt = f"""Extract the company name from this question.
Question: {question}
Reply with just the company name."""
            company_name = llm.invoke(extract_prompt).content.strip()

            sections = tool_report(company_name)
            context = f"FULL REPORT DATA FOR {company_name}:\n\n"
            for topic, content in sections.items():
                context += f"=== {topic.upper()} ===\n{content}\n\n"

        else:
            # SEARCH
            steps += 1
            tools_used.append("search_hr_documents")

            if len(question.split()) > 15:
                steps += 1
                tools_used.append("query_expansion")
                sub_prompt = f"""Break this complex question into 2 simpler search queries.
Question: {question}
Reply with just 2 queries, one per line."""
                sub_queries = llm.invoke(sub_prompt).content.strip().split("\n")
                sub_queries = [q.strip() for q in sub_queries if q.strip()][:2]

                all_context = []
                all_sources = []
                for sq in sub_queries:
                    ctx, srcs = tool_search(sq)
                    all_context.append(ctx)
                    all_sources.extend(srcs)
                context = "\n\n".join(all_context)
                sources = list(set(all_sources))
            else:
                context, sources = tool_search(question)

        # ── Step 3: Build history ──
        steps += 1
        history_text = ""
        if chat_history:
            history_text = "Previous conversation:\n"
            for h in chat_history[-3:]:
                role = h.get("role", "")
                content = h.get("content", "")
                if role == "human":
                    history_text += f"User: {content}\n"
                elif role == "assistant":
                    history_text += f"Assistant: {content[:150]}...\n"
            history_text += "\n"

        # ── Step 4: Generate answer ──
        steps += 1
        tools_used.append("llm_generation")

        uploaded_note = ""
        if uploaded_doc_name:
            uploaded_note = f"Note: User has uploaded '{uploaded_doc_name}' which is also available."

        answer_prompt = f"""You are an intelligent HR Policy Assistant.
{history_text}
{uploaded_note}
Use ONLY the context below to answer. If not found, say so clearly.
Present answers in a well-structured format with bullet points.

Context:
{context}

Question: {question}
Answer:"""

        answer = llm.invoke(answer_prompt).content

        # ── Step 5: Self-correction ──
        steps += 1
        tools_used.append("self_correction")
        answer = tool_self_correct(question, answer)

        # Add sources to answer if available
        sources_str = ""
        if sources:
            sources_str = "\n\n**Sources:** " + ", ".join(
                set([s.replace("_", " ").replace(".pdf", "") for s in sources])
            )

        return {
            "answer": answer + sources_str,
            "tools_used": list(dict.fromkeys(tools_used)),
            "steps": steps,
            "success": True
        }

    except Exception as e:
        return {
            "answer": f"I encountered an error: {str(e)}. Please try rephrasing.",
            "tools_used": tools_used,
            "steps": steps,
            "success": False
        }