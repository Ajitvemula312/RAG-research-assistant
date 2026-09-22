from app.agents.retriever import retrieve_evidence
from app.agents.state import ResearchState
from app.retrieval.hybrid import HybridRetriever


def research(state: ResearchState, retriever: HybridRetriever, top_k: int = 5) -> ResearchState:
    try:
        return retrieve_evidence(state, retriever, top_k)
    except Exception as exc:  # noqa: BLE001 - retrieval adapters may raise provider-specific errors
        return {"retrieved_documents": [], "retrieved_evidence": [], "errors": [f"retrieval failed: {exc}"]}