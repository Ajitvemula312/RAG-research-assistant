from app.evaluation.metrics import retrieval_precision, retrieval_recall


def test_retrieval_metrics_at_k():
    assert retrieval_recall(["a"], ["a", "b"]) == 1.0
    assert retrieval_precision(["a"], ["a", "b"]) == 0.5