import logging
import sys

from app.config import EVAL_K_VALUES
from app.pipeline import build_index
from app.ingestion.loaders import DocumentLoadError
from app.embeddings.embedding_model import EmbeddingModelError
from app.vectorstore.faiss_store import VectorStoreError
from app.retrieval.bm25_search import BM25Error
from app.retrieval.hybrid_search import hybrid_search
from app.generation.query_rewriter import rewrite_query
from app.evaluation.eval_dataset import EVALUATION_QUESTIONS
from app.evaluation.retrieval_metrics import evaluate_retrieval, print_metrics_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("evaluate")


def main():
    try:
        vectorstore, bm25, bm25_documents = build_index()
    except (DocumentLoadError, EmbeddingModelError, VectorStoreError, BM25Error) as exc:
        logger.error(exc)
        sys.exit(1)

    if not EVALUATION_QUESTIONS:
        print("app/evaluation/eval_dataset.py has no questions. Add some and rerun.")
        return

    def retrieve_fn(question, k):
        search_query = rewrite_query(question)
        return hybrid_search(vectorstore, bm25, bm25_documents, search_query, k)

    results = evaluate_retrieval(retrieve_fn, EVALUATION_QUESTIONS, EVAL_K_VALUES)
    print_metrics_table(results)


if __name__ == "__main__":
    main()
