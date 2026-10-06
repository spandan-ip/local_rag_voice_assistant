import logging

from langchain_core.documents import Document

from app.ingestion.loaders import load_document, DocumentLoadError
from app.ingestion.metadata import add_document_id

logger = logging.getLogger(__name__)


def load_all_sources(sources: list[dict]) -> list[Document]:
    all_documents: list[Document] = []

    for entry in sources:
        source_type = entry["source_type"]
        source = entry["source"]
        try:
            docs = load_document(source_type, source)
            docs = add_document_id(docs, source_type, source)
            all_documents.extend(docs)
            logger.info("Loaded %d document(s) from %s (%s)", len(docs), source, source_type)
        except DocumentLoadError as exc:
            logger.error("Skipping source '%s': %s", source, exc)

    if not all_documents:
        raise DocumentLoadError("No documents were loaded from any configured source.")

    return all_documents
