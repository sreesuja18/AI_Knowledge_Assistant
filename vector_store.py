from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import pickle

def load_documents():
    folder_path = "data"
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            print(f"Loading: {file}")
            loader = PyPDFLoader(os.path.join(folder_path, file))
            documents.extend(loader.load())
    return documents

docs = load_documents()
print(f"Total pages loaded: {len(docs)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(docs)
print(f"Total chunks: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.from_documents(chunks, embeddings)
vector_db.save_local("vector_db")

# Save raw chunk texts for BM25
chunk_texts = [c.page_content for c in chunks]
chunk_metas = [c.metadata for c in chunks]
with open("bm25_chunks.pkl", "wb") as f:
    pickle.dump({"texts": chunk_texts, "metas": chunk_metas}, f)

print("Vector DB and BM25 index saved successfully!")