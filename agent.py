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

# ── Tool 1: Search documents ──
def tool_search(query):
    results = agent_search(query)
    context = "\n\n".join([r.page_content for r in results])
    sources = list(set([
        os.path.basename(r.metadata.get("source", "Unknown"))
        for r in results
    ]))
    return context, sources

# ── Tool 2: Compare companies ──
def tool_compare(company_a, company_b, topic):
    def get_info(company):
        results = vector_db.similarity_search(
            f"{topic} policy at {company}", k=3
        )
        return "\n".join([r.page_content for r in results])
    return get_info(company_a), get_info(company_b)

# ── Tool 3: Generate report ──
def tool_report(company_name):
    topics = [
        "leave vacation sick casual",
        "compensation salary bonus",
        "parental maternity paternity",
        "work from home hybrid",
        "performance review appraisal",
        "learning development training"
    ]
    sections = {}
    for topic in topics:
        results = vector_db.similarity_search(
            f"{topic} at {company_name}", k=2
        )
        if results:
            sections[topic] = "\n".join([r.page_content[:300] for r in results])
    return sections

# ── Tool 4: Self-correction ──
def tool_self_correct(question, answer):
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

def run_agent(question: str, chat_history: list = []) -> dict:
    """
    Agentic AI — multi-step reasoning with tool selection.
    Step 1: Classify the question type
    Step 2: Use the right tool
    Step 3: Generate answer
    Step 4: Self-correct the answer
    """
    tools_used = []
    steps = 0

    try:
        # ── Step 1: Classify question ──
        steps += 1
        classify_prompt = f"""Classify this HR policy question into one of these types:
1. SEARCH - general policy question about one company
2. COMPARE - comparing two companies
3. REPORT - requesting full company summary
4. CALCULATE - asking about number of days or amounts

Question: {question}
Reply with just the type: SEARCH, COMPARE, REPORT, or CALCULATE"""

        classification = llm.invoke(classify_prompt).content.strip().upper()
        for qtype in ["SEARCH", "COMPARE", "REPORT", "CALCULATE"]:
            if qtype in classification:
                classification = qtype
                break
        else:
            classification = "SEARCH"

        # ── Step 2: Use the right tool ──
        context = ""
        sources = []
        extra_info = ""

        if classification == "COMPARE":
            steps += 1
            tools_used.append("compare_companies")

            # Extract company names
            extract_prompt = f"""From this question, extract:
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
            context = f"INFO ABOUT {company_a}:\n{info_a}\n\nINFO ABOUT {company_b}:\n{info_b}"
            extra_info = f"comparing {company_a} vs {company_b}"

        elif classification == "REPORT":
            steps += 1
            tools_used.append("generate_company_report")

            # Extract company name
            extract_prompt = f"""Extract the company name from this question.
Question: {question}
Reply with just the company name."""
            company_name = llm.invoke(extract_prompt).content.strip()

            sections = tool_report(company_name)
            context = f"FULL REPORT DATA FOR {company_name}:\n\n"
            for topic, content in sections.items():
                context += f"=== {topic.upper()} ===\n{content}\n\n"
            extra_info = f"full report for {company_name}"

        else:
            # SEARCH or CALCULATE
            steps += 1
            tools_used.append("search_hr_documents")

            # Multi-step: break into sub-questions if complex
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

        # ── Step 3: Build history context ──
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

        answer_prompt = f"""You are an intelligent HR Policy Assistant.
{history_text}
Use ONLY the context below to answer. If not found, say so clearly.
Present the answer in a well-structured format with bullet points where appropriate.

Context:
{context}

Question: {question}
Answer:"""

        answer = llm.invoke(answer_prompt).content

        # ── Step 5: Self-correction ──
        steps += 1
        tools_used.append("self_correction")
        answer = tool_self_correct(question, answer)

        return {
            "answer": answer,
            "tools_used": list(dict.fromkeys(tools_used)),
            "steps": steps,
            "success": True
        }

    except Exception as e:
        return {
            "answer": f"I encountered an error processing your request. Please try rephrasing. Error: {str(e)}",
            "tools_used": tools_used,
            "steps": steps,
            "success": False
        }