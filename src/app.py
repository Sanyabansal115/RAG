from .rag_chain import build_rag_chain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

# Load environment variables (like OPENAI_API_KEY)
load_dotenv()

def main():
    # Example: Load a PDF and create embeddings
    pdf_path = "example.pdf"  # replace with your file path
    reader = PdfReader(pdf_path)
    texts = [page.extract_text() for page in reader.pages if page.extract_text()]

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create a vectorstore (FAISS example)
    vectorstore = FAISS.from_texts(texts, embeddings)

    # Build RAG chain
    qa_chain = build_rag_chain(vectorstore)

    # Example query
    query = "What is this document about?"
    result = qa_chain.run(query)
    print("Answer:", result)

if __name__ == "__main__":
    main()
