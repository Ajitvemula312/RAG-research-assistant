from pathlib import Path

from app.config.settings import Settings
from app.ingestion.chunker import chunk_text
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.web_loader import load_web
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import VectorStore
from app.schemas.models import Evidence


class IngestionPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        config = settings or Settings()
        self.embedder = EmbeddingModel(config.embedding_model)
        self.store = VectorStore(config.chroma_path)

    def ingest_pdf(self, path: str | Path, source_url: str | None = None) -> list[Evidence]:
        chunks = [chunk for text, metadata in load_pdf(path, source_url) for chunk in chunk_text(text, metadata)]
        self._store(chunks)
        return chunks

    def ingest_url(self, url: str) -> list[Evidence]:
        text, metadata = load_web(url)
        chunks = chunk_text(text, metadata)
        self._store(chunks)
        return chunks

    def _store(self, chunks: list[Evidence]) -> None:
        self.store.upsert(chunks, self.embedder.encode([chunk.text for chunk in chunks]))