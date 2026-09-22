import time

from fastapi import FastAPI

from app.agents.graph import ResearchGraph
from app.api.chat import router as chat_router
from app.api.evaluation import router as evaluation_router
from app.api.health import router as health_router
from app.config.settings import Settings
from app.llm.llama import OllamaLLM
from app.memory.sqlite import ConversationStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector_store import VectorStore
from app.schemas.models import ResearchRequest, ResearchResponse


class ResearchService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        settings = self.settings
        self.memory = ConversationStore(settings.sqlite_path)
        documents = VectorStore(settings.chroma_path).records()
        self.graph = ResearchGraph(HybridRetriever(BM25Retriever(documents), DenseRetriever(documents)), OllamaLLM(settings.llm_model, settings.llm_temperature, settings.llm_timeout_seconds), settings.retrieval_top_k)

    def answer(self, request: ResearchRequest) -> ResearchResponse:
        started = time.perf_counter()
        state = self.graph.invoke(request.question, request.session_id)
        self.memory.append(request.session_id, "user", request.question)
        self.memory.append(request.session_id, "assistant", state.get("draft_answer", ""))
        evidence = state.get("retrieved_evidence", [])
        return ResearchResponse(answer=state.get("draft_answer", ""), citations=state.get("citations", []), sources=[item.metadata for item in evidence], verification=state["verification_result"], latency_ms=(time.perf_counter() - started) * 1000, model_version=state.get("model_version", self.settings.llm_model), retrieval_version=state.get("retrieval_version", "hybrid-v1"))

    def latest_evaluation(self) -> dict:
        return {"status": "no evaluation has been run", "dataset": self.settings.evaluation_dataset}


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="AI Research Assistant", version="0.1.0")
    app.state.service = ResearchService(settings or Settings())
    app.include_router(chat_router)
    app.include_router(health_router)
    app.include_router(evaluation_router)
    return app


app = create_app()