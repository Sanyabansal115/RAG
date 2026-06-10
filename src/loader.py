import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document   

def scan_pdfs(data_dir: str) -> List[str]:
    pdfs = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            if name.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, name))
    return sorted(pdfs)

def load_pdfs_to_documents(pdf_paths: List[str]) -> List[Document]:
    all_docs: List[Document] = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata.setdefault("source", os.path.basename(path))
        all_docs.extend(docs)
    return all_docs
