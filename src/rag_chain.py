from langchain_mistralai.chat_models import ChatMistralAI
from langchain.chains import RetrievalQA

def create_rag_chain(mistral_api_key: str, mistral_model: str, vectordb):
    llm = ChatMistralAI(api_key=mistral_api_key, model=mistral_model)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    rag = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    print("RAG chain successfully created")
    return rag
