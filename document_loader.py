from langchain_community.document_loaders import PyPDFLoader
import os

data_path = "data"

documents = []

for file in os.listdir(data_path):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(data_path, file))
        documents.extend(loader.load())

print("Documents loaded:", len(documents))