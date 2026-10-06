import logging

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_embedding_model_cache: HuggingFaceEmbeddings | None = None


class EmbeddingModelError(Exception):
    pass


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model_cache

    if _embedding_model_cache is not None:
        return _embedding_model_cache

    try:
        model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        raise EmbeddingModelError(f"Failed to load embedding model '{EMBEDDING_MODEL}': {exc}") from exc

    _embedding_model_cache = model
    return model
