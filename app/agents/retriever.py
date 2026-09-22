from app.agents.state import ResearchState
from app.retrieval.hybrid import HybridRetriever


def retrieve_evidence(state: ResearchState, retriever: HybridRetriever, top_k: int = 5) -> ResearchState:
    results = []
    seen = set()
    for query in state.get("search_queries", [state["user_question"]]):
        for evidence in retriever.search(query, top_k):
            if evidence.metadata.chunk_id not in seen:
                results.append(evidence)
                seen.add(evidence.metadata.chunk_id)
    return {"retrieved_documents": results, "retrieved_evidence": results, "retrieval_attempts": state.get("retrieval_attempts", 0) + 1}