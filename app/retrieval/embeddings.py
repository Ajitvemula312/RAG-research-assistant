from collections.abc import Iterable


class EmbeddingModel:
    """Lazy Sentence Transformer wrapper with a deterministic offline fallback."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: str | Iterable[str]) -> list[list[float]]:
        values = [texts] if isinstance(texts, str) else list(texts)
        try:
            return self._load().encode(values, normalize_embeddings=True).tolist()
        except (ImportError, OSError):
            vectors = []
            for value in values:
                vector = [0.0] * 64
                for token in value.lower().split():
                    vector[hash(token) % len(vector)] += 1.0
                norm = sum(component * component for component in vector) ** 0.5 or 1.0
                vectors.append([component / norm for component in vector])
            return vectors