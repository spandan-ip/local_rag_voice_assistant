import os
import re
import pickle
import logging

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_BM25_FILENAME = "bm25_index.pkl"

# A small, common English stop word list. These carry little standalone
# meaning (they're grammatical glue, not content) so BM25 ignores them --
# otherwise every document would score high on "the", "is", "a", etc.
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "and", "or", "but", "if", "then",
    "so", "as", "it", "its", "this", "that", "these", "those", "with",
    "from", "by", "about", "into", "through", "during", "before", "after",
    "above", "below", "up", "down", "out", "off", "over", "under", "again",
    "further", "than", "too", "very", "can", "will", "just", "do", "does",
    "did", "doing", "not", "no", "nor", "only", "own", "same", "such",
    "he", "she", "they", "we", "you", "i", "his", "her", "their", "our",
    "your", "them", "what", "which", "who", "whom",
}


class BM25Error(Exception):
    pass


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOP_WORDS]


def _bm25_path(index_path: str) -> str:
    return os.path.join(index_path, _BM25_FILENAME)


def build_or_load_bm25(chunks, index_path: str, force_rebuild: bool = False):
    """Return (bm25_index, documents), building only if needed.

    Mirrors build_or_load_vectorstore in faiss_store.py: if a saved
    index already exists on disk, load it instead of re-tokenizing and
    re-fitting BM25 over every chunk again.
    """
    os.makedirs(index_path, exist_ok=True)
    path = _bm25_path(index_path)

    if not force_rebuild and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                bm25, documents = pickle.load(f)
            return bm25, documents
        except Exception as exc:
            logger.warning("Failed to load BM25 index (%s). Rebuilding.", exc)

    if not chunks:
        raise BM25Error("No chunks available to build a BM25 index, and no saved index found.")

    tokenized_corpus = [_tokenize(c.page_content) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    try:
        with open(path, "wb") as f:
            pickle.dump((bm25, chunks), f)
    except Exception as exc:
        logger.warning("Failed to save BM25 index: %s", exc)

    return bm25, chunks


def bm25_search(bm25: BM25Okapi, documents, query: str, k: int):
    if not query or not query.strip():
        raise BM25Error("Cannot search with an empty query.")

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_indices = ranked_indices[:k]

    return [documents[i] for i in top_indices]
