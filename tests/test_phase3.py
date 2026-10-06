from app.evaluation.retrieval_metrics import precision_at_k, recall_at_k, hit_rate_at_k


def test_precision_at_k_all_relevant():
    retrieved = ["a.pdf", "b.pdf", "c.pdf"]
    relevant = {"a.pdf", "b.pdf", "c.pdf"}
    assert precision_at_k(retrieved, relevant, k=3) == 1.0


def test_precision_at_k_none_relevant():
    retrieved = ["a.pdf", "b.pdf", "c.pdf"]
    relevant = {"x.pdf"}
    assert precision_at_k(retrieved, relevant, k=3) == 0.0


def test_precision_at_k_partial():
    retrieved = ["a.pdf", "x.pdf", "b.pdf", "y.pdf"]
    relevant = {"a.pdf", "b.pdf"}
    # top 4: a.pdf, x.pdf, b.pdf, y.pdf -> 2 relevant out of 4
    assert precision_at_k(retrieved, relevant, k=4) == 0.5


def test_recall_at_k_finds_all_relevant():
    retrieved = ["a.pdf", "b.pdf", "c.pdf"]
    relevant = {"a.pdf", "b.pdf"}
    assert recall_at_k(retrieved, relevant, k=3) == 1.0


def test_recall_at_k_finds_none():
    retrieved = ["x.pdf", "y.pdf"]
    relevant = {"a.pdf", "b.pdf"}
    assert recall_at_k(retrieved, relevant, k=2) == 0.0


def test_recall_at_k_partial():
    retrieved = ["a.pdf", "x.pdf"]
    relevant = {"a.pdf", "b.pdf"}
    # found 1 of 2 relevant docs
    assert recall_at_k(retrieved, relevant, k=2) == 0.5


def test_recall_at_k_no_relevant_docs_defined():
    assert recall_at_k(["a.pdf"], set(), k=1) == 0.0


def test_hit_rate_at_k_hit():
    retrieved = ["x.pdf", "a.pdf", "y.pdf"]
    relevant = {"a.pdf"}
    assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0


def test_hit_rate_at_k_miss():
    retrieved = ["x.pdf", "y.pdf"]
    relevant = {"a.pdf"}
    assert hit_rate_at_k(retrieved, relevant, k=2) == 0.0


def test_hit_rate_at_k_only_checks_top_k():
    # relevant doc is retrieved, but outside the top-1 window
    retrieved = ["x.pdf", "a.pdf"]
    relevant = {"a.pdf"}
    assert hit_rate_at_k(retrieved, relevant, k=1) == 0.0
