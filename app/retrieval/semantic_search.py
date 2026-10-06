from app.vectorstore.faiss_store import similarity_search


def semantic_search(vectorstore, query: str, k: int):
    return similarity_search(vectorstore, query, k)
