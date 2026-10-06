import re

_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "and", "or", "but", "if", "then",
    "so", "as", "it", "its", "this", "that", "these", "those", "with",
    "from", "by", "about", "into", "through", "during", "before", "after",
    "not", "no", "nor", "only", "own", "same", "such", "what", "which",
}


def _significant_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 2}


def filter_relevant_chunks(chunks: list, answer_text: str, min_overlap: float = 0.12) -> list:
    """Keep only the chunks that plausibly contributed to the answer.

    Hybrid retrieval (FAISS + BM25 + RRF) sometimes pulls in a chunk
    that scores reasonably on its own but that the LLM ends up not
    actually using -- e.g. a chunk from a different document that
    happens to share a keyword with the question. Listing it as a
    "source" would then be misleading. This keeps a chunk only if a
    meaningful fraction of its significant (non-stop) words also show
    up in the final answer text.

    This is a simple heuristic, not a guarantee -- it can't tell
    whether a chunk was used for reasoning versus just phrasing. If
    filtering would remove every chunk, the original unfiltered list is
    returned instead, since a well-grounded answer should always show
    at least the sources it came from.
    """
    if not chunks or not answer_text or not answer_text.strip():
        return chunks

    answer_words = _significant_words(answer_text)
    if not answer_words:
        return chunks

    kept = []
    for chunk in chunks:
        chunk_words = _significant_words(chunk.page_content)
        if not chunk_words:
            continue
        overlap = len(chunk_words & answer_words) / len(chunk_words)
        if overlap >= min_overlap:
            kept.append(chunk)

    return kept if kept else chunks
