import hashlib

from langchain_core.documents import Document


def generate_document_id(source_type: str, source: str) -> str:
    raw = f"{source_type}:{source}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def add_document_id(documents: list[Document], source_type: str, source: str) -> list[Document]:
    doc_id = generate_document_id(source_type, source)
    for doc in documents:
        doc.metadata["document_id"] = doc_id
    return documents
