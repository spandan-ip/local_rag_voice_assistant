def _doc_key(doc):
    # Identifies "the same chunk" across two independently-produced ranked
    # lists (semantic vs BM25), even though they may be different Document
    # object instances (e.g. one loaded from a pickle, one from FAISS).
    return (doc.metadata.get("document_id"), doc.metadata.get("chunk_index"))


def reciprocal_rank_fusion(ranked_lists: list[list], k_constant: int = 60) -> list:
    """Merge multiple ranked lists of Documents into one fused ranking.

    RRF_score(doc) = sum over each list of 1 / (k_constant + rank_in_that_list)

    A document that ranks well in BOTH lists (even if not #1 in either)
    scores higher than a document that ranks #1 in only one list -- this
    is what makes fusion more robust than trusting either method alone.
    """
    scores: dict = {}
    doc_lookup: dict = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_constant + rank)
            doc_lookup[key] = doc

    ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [doc_lookup[key] for key in ranked_keys]
