from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100

def split_documents(documents: List[Document], target_chunks: int = 12,
                    chunk_size: int = DEFAULT_CHUNK_SIZE,
                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if len(chunks) == target_chunks:
        return chunks

    total_chars = sum(len(c.page_content) for c in chunks) or 1
    avg_target = max(100, int(total_chars / target_chunks))
    adjusted_splitter = RecursiveCharacterTextSplitter(
        chunk_size=avg_target,
        chunk_overlap=min(chunk_overlap, max(0, avg_target // 10)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    adjusted = adjusted_splitter.split_documents(documents)

    if len(adjusted) < target_chunks:
        adjusted = adjusted.copy()
        while len(adjusted) < target_chunks and len(adjusted) > 0:
            idx = max(range(len(adjusted)), key=lambda i: len(adjusted[i].page_content))
            big = adjusted.pop(idx)
            mid = len(big.page_content) // 2
            left = Document(page_content=big.page_content[:mid], metadata=big.metadata)
            right = Document(page_content=big.page_content[mid:], metadata=big.metadata)
            adjusted.extend([left, right])
    elif len(adjusted) > target_chunks:
        adjusted = adjusted[:target_chunks]

    return adjusted
