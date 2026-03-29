from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

# Load vector DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)
client = Groq(api_key="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")

# 20 test questions with expected keywords
test_cases = [
    {"question": "What is the leave policy in TCS?", "keywords": ["privilege", "18", "sick", "casual"]},
    {"question": "What is the maternity leave in Cognizant?", "keywords": ["26 weeks", "maternity"]},
    {"question": "What is the paternity leave in Wipro?", "keywords": ["5 days", "paternity"]},
    {"question": "What is the paternity leave in Infosys?", "keywords": ["5 days", "paternity"]},
    {"question": "What is the hybrid work policy in Wipro?", "keywords": ["3 days", "hybrid"]},
    {"question": "What is the dress code policy?", "keywords": ["business casual", "professional"]},
    {"question": "What is the exit process for employees?", "keywords": ["resignation", "notice period"]},
    {"question": "What is the vacation policy at Tesla?", "keywords": ["10 days", "15 days", "vacation"]},
    {"question": "What is the parental leave at Google?", "keywords": ["18", "parental", "leave"]},
    {"question": "What is the hybrid work policy at Google?", "keywords": ["3 days", "50 miles"]},
    {"question": "What is the sick leave policy in TCS?", "keywords": ["12 days", "sick"]},
    {"question": "What is the bereavement leave at J&J?", "keywords": ["10 days", "bereavement"]},
    {"question": "What is the performance review system in Infosys?", "keywords": ["icount", "exceeds", "annual"]},
    {"question": "What is the learning platform in TCS?", "keywords": ["ievolve", "learning"]},
    {"question": "What is the provident fund contribution in Cognizant?", "keywords": ["12%", "provident"]},
    {"question": "What is the notice period for employees?", "keywords": ["30", "90", "notice"]},
    {"question": "What is the Amazon return to office policy?", "keywords": ["5 days", "january 2025"]},
    {"question": "What is the Microsoft parental leave policy?", "keywords": ["20 weeks", "parental"]},
    {"question": "What is the J&J credo?", "keywords": ["1943", "patients", "shareholders"]},
    {"question": "What is the Career Choice programme at Amazon?", "keywords": ["95%", "tuition", "90 days"]},
]

def get_answer(question):
    results = vector_db.similarity_search(question, k=5)
    context = "\n\n".join([doc.page_content for doc in results])
    prompt = f"""You are a company policy assistant. Answer using ONLY the context below.
If the answer is not in the context, say: "I could not find this information in the company documents."

Context:
{context}

Question: {question}
Answer:"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.lower()

# Run evaluation
print("=" * 65)
print("        RAG SYSTEM EVALUATION REPORT")
print("=" * 65)

correct = 0

for i, test in enumerate(test_cases):
    answer = get_answer(test["question"])
    matched = any(kw.lower() in answer for kw in test["keywords"])
    status = "PASS" if matched else "FAIL"
    if matched:
        correct += 1
    print(f"Q{i+1:02d}: {test['question'][:52]:<52} [{status}]")

accuracy = (correct / len(test_cases)) * 100

print("=" * 65)
print(f"  TOTAL QUESTIONS : {len(test_cases)}")
print(f"  CORRECT         : {correct}")
print(f"  INCORRECT       : {len(test_cases) - correct}")
print(f"  ACCURACY        : {accuracy:.1f}%")
print("=" * 65)