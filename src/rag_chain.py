from dataclasses import dataclass
from typing import Any, Dict, List

import requests

from .loader import Document


MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


@dataclass
class SimpleRAGChain:
    mistral_api_key: str
    mistral_model: str
    vectordb: Any
    top_k: int = 3

    def _format_context(self, documents: List[Document]) -> str:
        sections: List[str] = []
        for index, document in enumerate(documents, start=1):
            metadata = document.metadata or {}
            source = metadata.get("source", "unknown source")
            page = metadata.get("page")
            header = f"[{index}] Source: {source}"
            if page is not None:
                header += f" | Page: {page}"
            sections.append(f"{header}\n{document.page_content}")
        return "\n\n".join(sections)

    def invoke(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        query = inputs.get("query", "").strip()
        if not query:
            return {"result": "", "source_documents": []}

        source_documents = self.vectordb.similarity_search(query, k=self.top_k)
        if not source_documents:
            return {
                "result": "I could not find relevant information in the provided documents.",
                "source_documents": [],
            }

        context = self._format_context(source_documents)

        messages = [
            {
                "role": "system",
                "content": (
                    "You answer questions using only the provided document context. "
                    "If the answer is not in the context, say that you could not find it. "
                    "Keep the response concise and factual."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]

        response = requests.post(
            MISTRAL_API_URL,
            headers={
                "Authorization": f"Bearer {self.mistral_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.mistral_model,
                "messages": messages,
                "temperature": 0.2,
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        answer = payload["choices"][0]["message"]["content"].strip()
        return {"result": answer, "source_documents": source_documents}

    def run(self, query: str) -> str:
        return self.invoke({"query": query})["result"]


def create_rag_chain(mistral_api_key: str, mistral_model: str, vectordb, top_k: int = 3):
    print("RAG chain successfully created")
    return SimpleRAGChain(
        mistral_api_key=mistral_api_key,
        mistral_model=mistral_model,
        vectordb=vectordb,
        top_k=top_k,
    )


build_rag_chain = create_rag_chain
