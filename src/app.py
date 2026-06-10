import os

from dotenv import load_dotenv

from .chunker import split_documents
from .loader import load_pdfs_to_documents, scan_pdfs
from .rag_chain import create_rag_chain
from .utils import load_env
from .vectorstore_build import DEFAULT_STORE_FILENAME, build_vectorstore, load_vectorstore


def _load_or_build_vectorstore(chunks, persist_directory: str):
    store_path = os.path.join(persist_directory, DEFAULT_STORE_FILENAME)
    if os.path.exists(store_path):
        try:
            vectordb = load_vectorstore(persist_directory=persist_directory)
            if vectordb.count() > 0:
                return vectordb
        except Exception:
            pass

    return build_vectorstore(
        chunks=chunks,
        persist_directory=persist_directory,
    )


def main():
    load_dotenv()
    config = load_env()

    if not config["MISTRAL_API_KEY"]:
        raise RuntimeError("MISTRAL_API_KEY is required. Add it to the .env file.")

    data_dir = os.path.join(os.getcwd(), "data")
    persist_directory = os.path.join(os.getcwd(), "chroma_db")

    pdf_paths = scan_pdfs(data_dir)
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {data_dir}. Add your requirements documents to the data folder."
        )

    documents = load_pdfs_to_documents(pdf_paths)
    chunks = split_documents(documents)
    vectordb = _load_or_build_vectorstore(chunks, persist_directory)
    qa_chain = create_rag_chain(config["MISTRAL_API_KEY"], config["MISTRAL_MODEL"], vectordb)

    print("ReqMind is ready. Ask a question or type 'exit' to quit.")

    while True:
        try:
            query = input("\nQuestion: ").strip()
        except EOFError:
            print("\nGoodbye.")
            break
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not query:
            continue

        result = qa_chain.invoke({"query": query})
        answer = result.get("result", "")
        sources = result.get("source_documents", [])

        print("\nAnswer:")
        print(answer)

        if sources:
            print("\nSources:")
            for index, doc in enumerate(sources, start=1):
                source_name = doc.metadata.get("source", "unknown")
                preview = doc.page_content.replace("\n", " ")[:200]
                print(f"{index}. {source_name}: {preview}")


if __name__ == "__main__":
    main()
