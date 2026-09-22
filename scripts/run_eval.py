import argparse
import json

from app.evaluation.evaluator import evaluate
from app.evaluation.regression import check_regression
from app.main import ResearchService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/evaluation/questions.jsonl")
    parser.add_argument("--baseline", default="data/evaluation/baseline.json")
    args = parser.parse_args()
    service = ResearchService()
    metrics = evaluate(service.graph, args.dataset, service.settings.mlflow_uri)
    print(json.dumps(metrics, indent=2))
    failures = check_regression(metrics, args.baseline)
    if failures:
        raise SystemExit("Regression detected: " + ", ".join(failures))


if __name__ == "__main__":
    main()