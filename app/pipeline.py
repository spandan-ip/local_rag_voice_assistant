from app.config import VECTOR_STORE_DIR
from app.ingestion.source_registry import get_all_sources
from app.ingestion.ingestion_pipeline import load_all_sources
from app.chunking.splitter import chunk_documents
from app.embeddings.embedding_model import get_embedding_model
from app.vectorstore.faiss_store import build_or_load_vectorstore
from app.retrieval.bm25_search import build_or_load_bm25


def build_index(force_rebuild: bool = False):
    documents = load_all_sources(get_all_sources())
    chunks = chunk_documents(documents)

    # Tag each chunk with its position in the corpus. This is how RRF
    # later recognizes "the same chunk" when it shows up in both the
    # FAISS ranking and the BM25 ranking.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    embedding_model = get_embedding_model()
    vectorstore = build_or_load_vectorstore(chunks, embedding_model, VECTOR_STORE_DIR, force_rebuild=force_rebuild)
    bm25, bm25_documents = build_or_load_bm25(chunks, VECTOR_STORE_DIR, force_rebuild=force_rebuild)

    return vectorstore, bm25, bm25_documents
