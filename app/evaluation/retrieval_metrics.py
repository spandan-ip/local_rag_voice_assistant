def precision_at_k(retrieved_sources: list[str], relevant_sources: set[str], k: int) -> float:
    # Of the K documents we retrieved, what fraction were actually relevant?
    if k == 0:
        return 0.0
    top_k = retrieved_sources[:k]
    hits = sum(1 for s in top_k if s in relevant_sources)
    return hits / k


def recall_at_k(retrieved_sources: list[str], relevant_sources: set[str], k: int) -> float:
    # Of all the documents that are actually relevant, what fraction did we find in the top K?
    if not relevant_sources:
        return 0.0
    top_k = retrieved_sources[:k]
    hits = sum(1 for s in top_k if s in relevant_sources)
    return hits / len(relevant_sources)


def hit_rate_at_k(retrieved_sources: list[str], relevant_sources: set[str], k: int) -> float:
    # Did we find AT LEAST ONE relevant document in the top K? (1 or 0, per question)
    top_k = retrieved_sources[:k]
    return 1.0 if any(s in relevant_sources for s in top_k) else 0.0


def evaluate_retrieval(retrieve_fn, eval_dataset: list[dict], k_values: list[int]) -> dict:
    """Run every question in eval_dataset through retrieve_fn, for every K in k_values.

    retrieve_fn: a function (query: str, k: int) -> list[Document]. This is
    kept generic (rather than hardcoding a vectorstore) so the SAME metrics
    code can evaluate semantic-only search, or hybrid search, or any future
    retrieval method -- whatever retrieve_fn you pass in.

    Returns: {k: {"precision": float, "recall": float, "hit_rate": float}}
    Each value is averaged across all questions in eval_dataset.
    """
    results = {}

    for k in k_values:
        precisions, recalls, hits = [], [], []

        for item in eval_dataset:
            question = item["question"]
            relevant_sources = set(item["relevant_sources"])

            retrieved_docs = retrieve_fn(question, k)
            retrieved_sources = [doc.metadata.get("source", "") for doc in retrieved_docs]

            precisions.append(precision_at_k(retrieved_sources, relevant_sources, k))
            recalls.append(recall_at_k(retrieved_sources, relevant_sources, k))
            hits.append(hit_rate_at_k(retrieved_sources, relevant_sources, k))

        results[k] = {
            "precision": sum(precisions) / len(precisions),
            "recall": sum(recalls) / len(recalls),
            "hit_rate": sum(hits) / len(hits),
        }

    return results


def print_metrics_table(results: dict) -> None:
    print(f"{'K':<6}{'Precision':<12}{'Recall':<12}{'Hit Rate':<12}")
    for k, m in results.items():
        print(f"{k:<6}{m['precision']:<12.2f}{m['recall']:<12.2f}{m['hit_rate']:<12.2f}")
