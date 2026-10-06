from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.retrieval.bm25_search import _tokenize, bm25_search, BM25Error
from app.retrieval.rrf import reciprocal_rank_fusion


def _make_doc(text, document_id, chunk_index):
    return Document(
        page_content=text,
        metadata={"document_id": document_id, "chunk_index": chunk_index, "source": f"{document_id}.pdf"},
    )


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def test_tokenize_lowercases_and_removes_stop_words():
    tokens = _tokenize("The Quick Brown Fox is in the Garden")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "quick" in tokens
    assert "brown" in tokens
    assert "garden" in tokens


def test_bm25_search_ranks_matching_document_first():
    docs = [
        _make_doc("The company offers 20 days of annual leave per year", "policy", 0),
        _make_doc("Employees must submit expense reports within 30 days", "policy", 1),
        _make_doc("The office kitchen is on the third floor", "policy", 2),
    ]
    tokenized_corpus = [_tokenize(d.page_content) for d in docs]
    bm25 = BM25Okapi(tokenized_corpus)

    results = bm25_search(bm25, docs, "annual leave days", k=1)

    assert len(results) == 1
    assert "leave" in results[0].page_content.lower()


def test_bm25_search_rejects_empty_query():
    docs = [_make_doc("some text", "a", 0)]
    bm25 = BM25Okapi([_tokenize(docs[0].page_content)])

    try:
        bm25_search(bm25, docs, "   ", k=1)
        assert False, "expected BM25Error"
    except BM25Error:
        pass


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------

def test_rrf_ranks_document_found_by_both_methods_highest():
    doc_a = _make_doc("a", "a", 0)
    doc_b = _make_doc("b", "b", 0)
    doc_c = _make_doc("c", "c", 0)

    semantic_ranked = [doc_a, doc_b, doc_c]  # a is #1 semantically
    bm25_ranked = [doc_b, doc_a, doc_c]      # b is #1 by keyword, a is #2

    fused = reciprocal_rank_fusion([semantic_ranked, bm25_ranked], k_constant=60)

    fused_ids = [d.metadata["document_id"] for d in fused]
    # a and b both rank well in both lists; c ranks last in both -> c should be last.
    assert fused_ids[-1] == "c"
    assert set(fused_ids[:2]) == {"a", "b"}


def test_rrf_deduplicates_same_chunk_across_lists():
    doc = _make_doc("shared chunk", "x", 0)
    only_in_other = _make_doc("other chunk", "y", 0)

    fused = reciprocal_rank_fusion([[doc], [doc, only_in_other]], k_constant=60)

    keys = [(d.metadata["document_id"], d.metadata["chunk_index"]) for d in fused]
    assert len(keys) == len(set(keys)), "same chunk should not appear twice in fused results"
    assert len(fused) == 2
