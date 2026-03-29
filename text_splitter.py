from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_documents():

    folder_path = "data"
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            documents.extend(loader.load())

    return documents


# Load documents
docs = load_documents()

# Split into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

print("Total chunks created:", len(chunks))
print(chunks[0].page_content[:300])