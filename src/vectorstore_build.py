import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document

def build_vectorstore(chunks: List[Document],
                      persist_directory: str = "./chroma_db",
                      collection_name: str = "rag_studentid"):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    os.makedirs(persist_directory, exist_ok=True)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    vectordb.persist()
    print("Vector store ready for retrieval")
    return vectordb
