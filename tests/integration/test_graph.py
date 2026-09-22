from app.agents.graph import ResearchGraph
from app.llm.llama import DeterministicLLM
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.schemas.models import Evidence, SourceMetadata


def test_graph_returns_structured_insufficient_evidence():
    documents = [Evidence(text="The study reports calibrated uncertainty.", metadata=SourceMetadata(document_id="d1", chunk_id="d1:0"))]
    graph = ResearchGraph(HybridRetriever(BM25Retriever(documents), DenseRetriever(documents)), DeterministicLLM())
    result = graph.invoke("What does the study report?")
    assert "verification_result" in result
    assert result["citations"][0].document_id == "d1"