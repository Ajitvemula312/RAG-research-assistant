import json
from pathlib import Path

from app.schemas.models import Evidence


class VectorStore:
    """Chroma-backed store with a JSON fallback useful for CI and unit tests."""

    def __init__(self, path: str = "data/processed/chroma", collection_name: str = "research") -> None:
        self.path = Path(path)
        self.collection_name = collection_name
        self._collection = None
        try:
            import chromadb

            self._collection = chromadb.PersistentClient(path=str(self.path)).get_or_create_collection(collection_name)
        except (ImportError, OSError, RuntimeError):
            self.path.mkdir(parents=True, exist_ok=True)

    def upsert(self, documents: list[Evidence], embeddings: list[list[float]]) -> None:
        if not documents:
            return
        if self._collection:
            self._collection.upsert(ids=[item.metadata.chunk_id for item in documents], documents=[item.text for item in documents], embeddings=embeddings, metadatas=[self._chroma_metadata(item) for item in documents])
            return
        records = {item.metadata.chunk_id: {"text": item.text, "metadata": item.metadata.model_dump()} for item in documents}
        self.path.mkdir(parents=True, exist_ok=True)
        existing = self._read_fallback()
        existing.update(records)
        (self.path / "records.json").write_text(json.dumps(existing), encoding="utf-8")

    def query(self, embedding: list[float], top_k: int = 5) -> list[Evidence]:
        if self._collection:
            result = self._collection.query(query_embeddings=[embedding], n_results=top_k)
            return [Evidence(text=text, metadata=self._source_metadata(metadata), score=1 - distance, retrieval_method="dense") for text, metadata, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True)]
        return []

    def _read_fallback(self) -> dict:
        file = self.path / "records.json"
        return json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}

    def records(self) -> list[Evidence]:
        if self._collection:
            result = self._collection.get(include=["documents", "metadatas"])
            return [Evidence(text=text, metadata=self._source_metadata(metadata)) for text, metadata in zip(result["documents"], result["metadatas"], strict=True)]
        return [Evidence(text=value["text"], metadata=value["metadata"]) for value in self._read_fallback().values()]

    @staticmethod
    def _chroma_metadata(item: Evidence) -> dict[str, str | int | float | bool]:
        metadata = item.metadata.model_dump(exclude_none=True)
        authors = metadata.get("authors")
        if isinstance(authors, list):
            metadata["authors"] = ", ".join(authors) or "unknown"
        return {key: value for key, value in metadata.items() if value is not None}

    @staticmethod
    def _source_metadata(metadata: dict) -> dict:
        normalized = dict(metadata)
        authors = normalized.get("authors")
        if isinstance(authors, str):
            normalized["authors"] = [] if authors == "unknown" else [author.strip() for author in authors.split(",") if author.strip()]
        return normalized