import os
import pickle
from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .loader import Document


DEFAULT_STORE_FILENAME = "vectorstore.pkl"
DEFAULT_TEXT_FEATURES = {
    "lowercase": True,
    "stop_words": "english",
    "ngram_range": (1, 2),
}


@dataclass
class VectorStore:
    documents: List[Document]
    vectorizer: TfidfVectorizer
    matrix: object
    persist_path: str

    def count(self) -> int:
        return len(self.documents)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        with open(self.persist_path, "wb") as file_handle:
            pickle.dump(self, file_handle)

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        if not self.documents:
            return []

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.matrix).ravel()
        if similarities.size == 0 or float(similarities.max()) <= 0.0:
            return []

        top_indices = similarities.argsort()[::-1][:k]
        results: List[Document] = []
        for index in top_indices:
            if similarities[index] <= 0.0:
                continue
            results.append(self.documents[index])
        return results


def _store_path(persist_directory: str) -> str:
    return os.path.join(persist_directory, DEFAULT_STORE_FILENAME)


def load_vectorstore(persist_directory: str = "./chroma_db") -> VectorStore:
    persist_path = _store_path(persist_directory)
    with open(persist_path, "rb") as file_handle:
        return pickle.load(file_handle)


def build_vectorstore(chunks: List[Document], persist_directory: str = "./chroma_db") -> VectorStore:
    if not chunks:
        raise ValueError("Cannot build a vector store without any document chunks.")

    os.makedirs(persist_directory, exist_ok=True)
    texts = [chunk.page_content for chunk in chunks]
    vectorizer = TfidfVectorizer(**DEFAULT_TEXT_FEATURES)
    matrix = vectorizer.fit_transform(texts)

    store = VectorStore(
        documents=list(chunks),
        vectorizer=vectorizer,
        matrix=matrix,
        persist_path=_store_path(persist_directory),
    )
    store.save()
    print("Vector store ready for retrieval")
    return store
