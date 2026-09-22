import json
from pathlib import Path


def check_regression(metrics: dict[str, float], baseline_path: str, thresholds: dict[str, float] | None = None) -> list[str]:
    baseline = json.loads(Path(baseline_path).read_text()) if Path(baseline_path).exists() else {}
    thresholds = thresholds or {}
    failures = []
    for metric in ("recall_at_5", "groundedness", "citation_correctness"):
        minimum = thresholds.get(f"min_{metric}", baseline.get(metric, 0.0))
        if metrics.get(metric, 0.0) < minimum:
            failures.append(f"{metric} below threshold")
    if metrics.get("latency_ms", 0.0) > thresholds.get("max_latency_ms", float("inf")):
        failures.append("latency above threshold")
    if metrics.get("failure_rate", 0.0) > thresholds.get("max_failure_rate", 1.0):
        failures.append("failure rate above threshold")
    return failures