import argparse
import json

from app.config.settings import Settings
from app.evaluation.dataset import load_dataset
from app.evaluation.metrics import retrieval_precision, retrieval_recall
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=Settings().evaluation_dataset)
    args = parser.parse_args()
    documents = VectorStore(Settings().chroma_path).records()
    retrievers = {"bm25": BM25Retriever(documents), "dense": DenseRetriever(documents), "hybrid": HybridRetriever(BM25Retriever(documents), DenseRetriever(documents))}
    records = load_dataset(args.dataset)
    output = {}
    for name, retriever in retrievers.items():
        scores = [(retrieval_recall(record.expected_sources, [item.metadata.document_id for item in retriever.search(record.question)]), retrieval_precision(record.expected_sources, [item.metadata.document_id for item in retriever.search(record.question)])) for record in records]
        output[name] = {"recall_at_5": sum(item[0] for item in scores) / (len(scores) or 1), "precision_at_5": sum(item[1] for item in scores) / (len(scores) or 1)}
    baseline = output["bm25"]["recall_at_5"]
    output["hybrid_improvement_percent"] = ((output["hybrid"]["recall_at_5"] - baseline) / baseline * 100) if baseline else None
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()