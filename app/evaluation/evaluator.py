import time

from app.agents.graph import ResearchGraph
from app.evaluation.dataset import load_dataset
from app.evaluation.metrics import keyword_score, retrieval_precision, retrieval_recall


def evaluate(graph: ResearchGraph, dataset_path: str, mlflow_uri: str | None = None) -> dict[str, float]:
    records = load_dataset(dataset_path)
    totals = {"recall_at_5": 0.0, "precision_at_5": 0.0, "answer_relevance": 0.0, "groundedness": 0.0, "citation_correctness": 0.0, "failure_rate": 0.0, "latency_ms": 0.0}
    for record in records:
        started = time.perf_counter()
        try:
            state = graph.invoke(record.question)
            source_ids = [item.metadata.document_id for item in state.get("retrieved_evidence", [])]
            totals["recall_at_5"] += retrieval_recall(record.expected_sources, source_ids)
            totals["precision_at_5"] += retrieval_precision(record.expected_sources, source_ids)
            totals["answer_relevance"] += keyword_score(state.get("draft_answer", ""), record.expected_answer_criteria)
            totals["groundedness"] += float(state.get("verification_result").supported)
            totals["citation_correctness"] += float(state.get("verification_result").citations_valid)
        except (KeyError, RuntimeError, TypeError, ValueError):
            totals["failure_rate"] += 1
        totals["latency_ms"] += (time.perf_counter() - started) * 1000
    count = len(records) or 1
    for key in totals:
        if key != "latency_ms":
            totals[key] /= count
    totals["latency_ms"] /= count
    if mlflow_uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(mlflow_uri)
            with mlflow.start_run(run_name="research-assistant-evaluation"):
                mlflow.log_params({"evaluation_dataset": dataset_path, "retrieval_version": "hybrid-v1"})
                mlflow.log_metrics(totals)
        except (ImportError, RuntimeError, OSError):
            totals["mlflow_logging_failed"] = 1.0
    return totals