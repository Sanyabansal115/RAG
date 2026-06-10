from dataclasses import dataclass, field
import os
from typing import Dict, List

from pypdf import PdfReader


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, object] = field(default_factory=dict)


def scan_pdfs(data_dir: str) -> List[str]:
    pdfs: List[str] = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            if name.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, name))
    return sorted(pdfs)


def load_pdfs_to_documents(pdf_paths: List[str]) -> List[Document]:
    all_docs: List[Document] = []
    for path in pdf_paths:
        reader = PdfReader(path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue
            all_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(path),
                        "path": path,
                        "page": page_number,
                    },
                )
            )
    return all_docs
