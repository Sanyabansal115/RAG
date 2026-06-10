# ReqMind - Requirements RAG Assistant

ReqMind is a small Retrieval-Augmented Generation (RAG) project for querying requirements PDFs. It scans documents from `data/`, splits them into chunks, embeds the chunks with Sentence Transformers, stores them in a local Chroma vector database, and uses Mistral to answer natural-language questions with supporting source documents.

## What this project does

- Loads PDF documents from the `data/` folder.
- Splits document text into manageable overlapping chunks.
- Builds and persists a local Chroma vector store in `chroma_db/`.
- Creates a retrieval-based QA chain with Mistral.
- Returns answers together with source documents for traceability.

## Project Layout

- `main.py` - Python entry point that launches the app.
- `src/loader.py` - discovers PDFs and loads them into LangChain documents.
- `src/chunker.py` - splits documents into chunks and normalizes the chunk count.
- `src/vectorstore_build.py` - embeds chunks and persists them to Chroma.
- `src/rag_chain.py` - builds the RetrievalQA chain with Mistral.
- `src/utils.py` - loads environment variables from `.env`.
- `src/app.py` - sample launcher logic included in the repo.
- `data/` - place your source PDFs here. A copy of `RequirementsDocument.pdf` is included.
- `chroma_db/` - persisted vector database files.
- `Analysis_Report.md` - architecture and design notes.
- `requirements.txt` - Python dependencies.

## How It Works

1. PDFs are discovered recursively under `data/`.
2. Each PDF is loaded into LangChain documents.
3. Documents are split using `RecursiveCharacterTextSplitter` with a default chunk size of 1000 and overlap of 100.
4. Chunks are embedded with `all-MiniLM-L6-v2` and saved to Chroma.
5. The retriever uses the top 3 chunks for each question.
6. Mistral generates the final response using the retrieved context.

## Requirements

- Python 3.10 or later is recommended.
- A valid Mistral API key.
- The required PDF files in the `data/` folder.

## Installation

Create and activate a virtual environment, then install dependencies:

```powershell
pip install -r requirements.txt
```

If you are using PowerShell, you can activate a local virtual environment like this:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\venv\Scripts\Activate.ps1
```

## Environment Variables

Create a `.env` file in the project root with the following values:

```env
FIRST_NAME=YourName
MISTRAL_API_KEY=your-mistral-api-key
MISTRAL_MODEL=mistral-large-latest
```

`FIRST_NAME` is currently used as a default profile value in the helper utilities. `MISTRAL_API_KEY` is required for the chat model. `MISTRAL_MODEL` can be changed if you want to use a different supported Mistral model.

## Running the Project

1. Put your PDF documents in `data/`.
2. Make sure `.env` is configured.
3. Run the entry point:

```powershell
python main.py
```

Note: `src/app.py` is a starter launcher and may need to be aligned with the current function names in `src/rag_chain.py` if you are wiring the end-to-end demo path yourself.

## Example Usage

The analysis notes included in `Analysis_Report.md` describe the expected demo flow and sample questions such as:

- What should be the maximum response time for the system?
- How many concurrent users must the system support during peak?
- Where is Centennial College?

## Design Choices

- Sentence Transformers are used for fast local embeddings.
- Chroma is used for local persistence and simple retrieval.
- Chunk overlap is used to preserve context across boundaries.
- Retrieval is limited to the top 3 chunks to keep prompts concise.
- `stuff` chain type is used for a simple baseline QA pipeline.

## Notes

- The repository includes a persisted `chroma_db/` directory so you can inspect the stored index.
- If you change the PDFs, rebuild the vector store so the retriever reflects the new content.
- `Analysis_Report.md` contains the assignment-oriented architecture summary and demo script.

## Troubleshooting

- If the app warns about a missing API key, verify `MISTRAL_API_KEY` in `.env`.
- If answers look stale, delete or rebuild `chroma_db/` after updating the documents.
- If a PDF is not being picked up, confirm it is inside `data/` and has a `.pdf` extension.

## Dependencies

The project depends on the following packages from `requirements.txt`:

- `langchain`
- `langchain-community`
- `langchain-mistralai`
- `chromadb`
- `sentence-transformers`
- `pypdf`
- `python-dotenv`

## Related Files

- `Analysis_Report.md` for the architecture and assignment notes.
- `requirements.txt` for the Python dependency list.
- `src/` for the modular RAG pipeline implementation.