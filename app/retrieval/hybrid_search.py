from app.config import RRF_K, HYBRID_CANDIDATE_K
from app.retrieval.semantic_search import semantic_search
from app.retrieval.bm25_search import bm25_search
from app.retrieval.rrf import reciprocal_rank_fusion


def hybrid_search(vectorstore, bm25, bm25_documents, query: str, k: int):
    semantic_results = semantic_search(vectorstore, query, HYBRID_CANDIDATE_K)
    keyword_results = bm25_search(bm25, bm25_documents, query, HYBRID_CANDIDATE_K)

    fused = reciprocal_rank_fusion([semantic_results, keyword_results], k_constant=RRF_K)
    return fused[:k]
