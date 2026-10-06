import argparse
import logging
import sys

from app.config import TOP_K, CONVERSATION_MEMORY_TURNS
from app.pipeline import build_index
from app.ingestion.loaders import DocumentLoadError
from app.embeddings.embedding_model import EmbeddingModelError
from app.vectorstore.faiss_store import VectorStoreError
from app.retrieval.bm25_search import BM25Error
from app.retrieval.hybrid_search import hybrid_search
from app.generation.llm import generate_answer, LLMError
from app.generation.query_rewriter import rewrite_query
from app.generation.relevance_filter import filter_relevant_chunks
from app.api.formatting import build_sources
from app.history.session_store import create_session, save_message, get_recent_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("main")


def answer_question(vectorstore, bm25, bm25_documents, question: str, session_id: str, k: int = TOP_K) -> str:
    search_query = rewrite_query(question)
    retrieved = hybrid_search(vectorstore, bm25, bm25_documents, search_query, k)

    save_message(session_id, "user", question)

    if not retrieved:
        answer = "I don't know based on the provided sources."
        save_message(session_id, "assistant", answer)
        return answer

    history = get_recent_context(session_id, max_turns=CONVERSATION_MEMORY_TURNS)
    answer = generate_answer(question, retrieved, history=history)

    relevant_chunks = filter_relevant_chunks(retrieved, answer)
    sources = build_sources(relevant_chunks)
    save_message(session_id, "assistant", answer, sources=sources)

    sources_block = "\n".join(
        f"{i}. {s['source']}, page {s['page']}" if s["page"] is not None else f"{i}. {s['source']}"
        for i, s in enumerate(sources, start=1)
    )
    return f"{answer}\n\nSources:\n{sources_block}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    try:
        vectorstore, bm25, bm25_documents = build_index(force_rebuild=args.rebuild)
    except (DocumentLoadError, EmbeddingModelError, VectorStoreError, BM25Error) as exc:
        logger.error(exc)
        sys.exit(1)

    session_id = create_session()
    print(f"\nSession: {session_id}")
    print("Index ready. Ask a question (type 'exit' to quit).\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        try:
            print(f"\n{answer_question(vectorstore, bm25, bm25_documents, question, session_id)}\n")
        except (VectorStoreError, BM25Error, LLMError) as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()
