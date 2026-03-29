from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

print("\nAI-Based Smart Knowledge Assistant for Companies")
print("Ask a question about company policies\n")

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector database
vector_db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

while True:

    query = input("Ask a question: ")

    if query.lower() == "exit":
        break

    # Search relevant documents
    docs = vector_db.similarity_search(query, k=5)

    keywords = ["leave", "leaving", "vacation", "policy", "pto", "sick", "parental"]

    best_sentence = ""

    for doc in docs:

        text = doc.page_content

        lines = text.split("\n")

        for line in lines:

            line = line.strip()

            if len(line) < 20:
                continue

            if any(word in line.lower() for word in keywords):
                best_sentence = line
                break

        if best_sentence:
            break

    if best_sentence == "":
        best_sentence = docs[0].page_content[:200]

    print("\nAnswer:\n")
    print(best_sentence)
    print("\n---------------------------------\n")