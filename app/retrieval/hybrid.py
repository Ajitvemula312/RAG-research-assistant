from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.schemas.models import Evidence


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, bm25_weight: float = 0.5, dense_weight: float = 0.5) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    @staticmethod
    def _normalize(results: list[Evidence]) -> dict[str, float]:
        if not results:
            return {}
        maximum = max(item.score for item in results) or 1.0
        return {item.metadata.chunk_id: item.score / maximum for item in results}

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        lexical = self.bm25.search(query, top_k)
        semantic = self.dense.search(query, top_k)
        by_id = {item.metadata.chunk_id: item for item in lexical + semantic}
        lexical_scores = self._normalize(lexical)
        semantic_scores = self._normalize(semantic)
        ranked = sorted(by_id.values(), key=lambda item: self.bm25_weight * lexical_scores.get(item.metadata.chunk_id, 0) + self.dense_weight * semantic_scores.get(item.metadata.chunk_id, 0), reverse=True)
        return [item.model_copy(update={"score": self.bm25_weight * lexical_scores.get(item.metadata.chunk_id, 0) + self.dense_weight * semantic_scores.get(item.metadata.chunk_id, 0), "retrieval_method": "hybrid"}) for item in ranked[:top_k]]