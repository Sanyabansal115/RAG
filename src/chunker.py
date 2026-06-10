from typing import List

from .loader import Document


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if not text:
        return []

    step = max(1, chunk_size - chunk_overlap)
    pieces: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            pieces.append(chunk)
        if end >= len(text):
            break
        start += step
    return pieces


def split_documents(documents: List[Document], target_chunks: int = 12,
                    chunk_size: int = DEFAULT_CHUNK_SIZE,
                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Document]:
    chunks: List[Document] = []
    for document in documents:
        split_texts = _split_text(document.page_content, chunk_size, chunk_overlap)
        for index, text in enumerate(split_texts):
            metadata = dict(document.metadata)
            metadata["chunk_index"] = index
            chunks.append(Document(page_content=text, metadata=metadata))

    if not chunks:
        return []

    if len(chunks) == target_chunks:
        return chunks

    total_chars = sum(len(chunk.page_content) for chunk in chunks) or 1
    avg_target = max(100, int(total_chars / target_chunks))
    adjusted_chunks: List[Document] = []

    for document in documents:
        split_texts = _split_text(document.page_content, avg_target, min(chunk_overlap, max(0, avg_target // 10)))
        for index, text in enumerate(split_texts):
            metadata = dict(document.metadata)
            metadata["chunk_index"] = index
            adjusted_chunks.append(Document(page_content=text, metadata=metadata))

    if len(adjusted_chunks) < target_chunks:
        adjusted_chunks = adjusted_chunks.copy()
        while len(adjusted_chunks) < target_chunks and adjusted_chunks:
            idx = max(range(len(adjusted_chunks)), key=lambda item: len(adjusted_chunks[item].page_content))
            largest = adjusted_chunks.pop(idx)
            midpoint = max(1, len(largest.page_content) // 2)
            left = Document(page_content=largest.page_content[:midpoint], metadata=dict(largest.metadata))
            right = Document(page_content=largest.page_content[midpoint:], metadata=dict(largest.metadata))
            adjusted_chunks.extend([left, right])
    elif len(adjusted_chunks) > target_chunks:
        adjusted_chunks = adjusted_chunks[:target_chunks]

    return adjusted_chunks
