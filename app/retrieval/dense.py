from app.retrieval.embeddings import EmbeddingModel
from app.schemas.models import Evidence


class DenseRetriever:
    def __init__(self, documents: list[Evidence] | None = None, embedding_model: EmbeddingModel | None = None) -> None:
        self.documents = documents or []
        self.embedding_model = embedding_model or EmbeddingModel()
        self._vectors = self.embedding_model.encode([item.text for item in self.documents]) if self.documents else []

    def index(self, documents: list[Evidence]) -> None:
        self.documents = list(documents)
        self._vectors = self.embedding_model.encode([item.text for item in self.documents])

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        if not query or top_k <= 0 or not self.documents:
            return []
        query_vector = self.embedding_model.encode(query)[0]
        scored = [(sum(a * b for a, b in zip(query_vector, vector, strict=True)), index) for index, vector in enumerate(self._vectors)]
        scored.sort(reverse=True)
        return [self.documents[index].model_copy(update={"score": score, "retrieval_method": "dense"}) for score, index in scored[:top_k]]