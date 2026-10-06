import os
import logging

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

_INDEX_FILENAME = "index.faiss"


class VectorStoreError(Exception):
    pass


def _index_exists(index_path: str) -> bool:
    return os.path.exists(os.path.join(index_path, _INDEX_FILENAME))


def build_or_load_vectorstore(
    chunks: list[Document],
    embedding_model: Embeddings,
    index_path: str,
    force_rebuild: bool = False,
) -> FAISS:
    os.makedirs(index_path, exist_ok=True)

    if not force_rebuild and _index_exists(index_path):
        try:
            return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
        except Exception as exc:
            logger.warning("Failed to load existing index (%s). Rebuilding.", exc)

    if not chunks:
        raise VectorStoreError("No chunks to build a vector store from, and no saved index found.")

    try:
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        vectorstore.save_local(index_path)
    except Exception as exc:
        raise VectorStoreError(f"Failed to build FAISS index: {exc}") from exc

    return vectorstore


def similarity_search(vectorstore: FAISS, query: str, k: int) -> list[Document]:
    if not query or not query.strip():
        raise VectorStoreError("Cannot search with an empty query.")

    try:
        results = vectorstore.similarity_search(query, k=k)
    except Exception as exc:
        raise VectorStoreError(f"Similarity search failed: {exc}") from exc

    return results
