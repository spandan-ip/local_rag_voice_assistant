import os
import logging

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, WebBaseLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentLoadError(Exception):
    pass


def _load_pdf(path: str) -> list[Document]:
    if not os.path.exists(path):
        raise DocumentLoadError(f"PDF file not found: {path}")

    if os.path.getsize(path) == 0:
        raise DocumentLoadError(f"PDF file is empty: {path}")

    try:
        docs = PyPDFLoader(path).load()
    except Exception as exc:
        raise DocumentLoadError(f"Failed to read PDF '{path}': {exc}") from exc

    if not docs:
        raise DocumentLoadError(f"PDF produced no readable text: {path}")

    filename = os.path.basename(path)
    for doc in docs:
        doc.metadata["source"] = filename
        doc.metadata["source_type"] = "pdf"
        if "page" in doc.metadata:
            doc.metadata["page"] = doc.metadata["page"] + 1

    logger.info("Loaded PDF '%s' with %d page(s)", filename, len(docs))
    return docs


def _load_txt(path: str) -> list[Document]:
    if not os.path.exists(path):
        raise DocumentLoadError(f"TXT file not found: {path}")

    if os.path.getsize(path) == 0:
        raise DocumentLoadError(f"TXT file is empty: {path}")

    try:
        docs = TextLoader(path, encoding="utf-8").load()
    except Exception as exc:
        raise DocumentLoadError(f"Failed to read TXT '{path}': {exc}") from exc

    filename = os.path.basename(path)
    for doc in docs:
        doc.metadata["source"] = filename
        doc.metadata["source_type"] = "txt"

    logger.info("Loaded TXT '%s'", filename)
    return docs


def _load_csv(path: str) -> list[Document]:
    if not os.path.exists(path):
        raise DocumentLoadError(f"CSV file not found: {path}")

    if os.path.getsize(path) == 0:
        raise DocumentLoadError(f"CSV file is empty: {path}")

    try:
        docs = CSVLoader(path).load()
    except Exception as exc:
        raise DocumentLoadError(f"Failed to read CSV '{path}': {exc}") from exc

    if not docs:
        raise DocumentLoadError(f"CSV has no rows: {path}")

    filename = os.path.basename(path)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = filename
        doc.metadata["source_type"] = "csv"
        doc.metadata["row"] = i + 1

    logger.info("Loaded CSV '%s' with %d row(s)", filename, len(docs))
    return docs


def _load_web(url: str) -> list[Document]:
    if not url.startswith(("http://", "https://")):
        raise DocumentLoadError(f"Invalid URL: {url}")

    try:
        docs = WebBaseLoader(url).load()
    except Exception as exc:
        raise DocumentLoadError(f"Failed to load website '{url}': {exc}") from exc

    if not docs or not docs[0].page_content.strip():
        raise DocumentLoadError(f"Website returned no readable content: {url}")

    for doc in docs:
        doc.metadata["source"] = url
        doc.metadata["source_type"] = "web"
        doc.metadata["url"] = url
        # WebBaseLoader may already set a "title" key; keep it if present.

    logger.info("Loaded web page '%s'", url)
    return docs


_LOADERS = {
    "pdf": _load_pdf,
    "txt": _load_txt,
    "csv": _load_csv,
    "web": _load_web,
}


def load_document(source_type: str, source: str) -> list[Document]:
    loader_fn = _LOADERS.get(source_type.lower())
    if loader_fn is None:
        raise DocumentLoadError(
            f"Unsupported source_type '{source_type}'. Supported: {list(_LOADERS.keys())}"
        )
    return loader_fn(source)
