# ReqMind - Requirements RAG System Analysis Report

Student Name: Sanya Bansal

## 1) Purpose
ReqMind is a lightweight Retrieval-Augmented Generation (RAG) assistant for requirements PDFs. It scans documents from `data/`, splits them into overlapping chunks, builds a local TF-IDF vector store, and uses the Mistral API to answer questions with source-backed context.

## 2) Current Architecture
- `main.py` launches the application.
- `src/app.py` orchestrates document loading, chunking, vector store creation, and the interactive Q&A loop.
- `src/loader.py` scans PDFs and extracts page text with `pypdf`.
- `src/chunker.py` performs chunking with a default size of 1000 and overlap of 100.
- `src/vectorstore_build.py` fits a TF-IDF model, persists the store to `chroma_db/vectorstore.pkl`, and supports cosine-similarity retrieval.
- `src/rag_chain.py` formats retrieved context and calls the Mistral chat completions API.
- `src/utils.py` loads `.env` settings such as `FIRST_NAME`, `MISTRAL_API_KEY`, and `MISTRAL_MODEL`.

## 3) Component Roles
- Loader: discover PDFs and extract text pages.
- Chunker: split long pages into manageable overlapping chunks.
- Vector Store: build and persist a searchable local index.
- Retriever: select the top 3 most relevant chunks for each query.
- Generator: use Mistral to produce a grounded answer from the retrieved context.
- CLI: present the answer and source previews in the terminal.

## 4) Design Decisions
- TF-IDF was chosen for simplicity, speed, and low operational overhead.
- Local pickle persistence was chosen so the repository stays self-contained.
- A top-k retrieval strategy keeps prompts short and predictable.
- The app returns a fallback message when no relevant chunks are found.
- The implementation avoids heavyweight chain frameworks and external vector DB dependencies.

## 5) Runbook
1. Install dependencies with `pip install -r requirements.txt`.
2. Configure `.env` with `MISTRAL_API_KEY` and `MISTRAL_MODEL`.
3. Place requirement PDFs in `data/`.
4. Run `python main.py`.
5. Ask a question, review the answer, and type `exit` to quit.

## 6) Demo Questions
- What should be the maximum response time for the system?
- How many concurrent users must the system support during peak?
- Where is Centennial College?

## 7) Notes
- The vector store artifact is stored locally under `chroma_db/vectorstore.pkl`.
- If the PDFs change, delete the pickle file to force a rebuild on the next run.
- The app is designed to work cleanly in the requested `AI` Conda environment.
