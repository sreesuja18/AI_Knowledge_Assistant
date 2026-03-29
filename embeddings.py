from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

print("Loading documents...")

data_path = "data"
documents = []

# Load all PDFs from data folder
for file in os.listdir(data_path):
    if file.endswith(".pdf"):
        print("Reading:", file)
        loader = PyPDFLoader(os.path.join(data_path, file))
        documents.extend(loader.load())

print("Total pages loaded:", len(documents))

print("Creating embeddings...")

# Lightweight embedding model (works on 4GB RAM)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

print("Creating FAISS vector database...")

# Create vector database directly
vector_db = FAISS.from_documents(documents, embeddings)

# Save vector database
vector_db.save_local("vector_db")

print("\nSUCCESS: vector_db folder created!")