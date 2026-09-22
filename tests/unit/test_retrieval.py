from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.schemas.models import Evidence, SourceMetadata


def documents():
    return [Evidence(text="uncertainty sensor fusion covariance", metadata=SourceMetadata(document_id="paper-1", chunk_id="paper-1:0")), Evidence(text="adverse weather perception", metadata=SourceMetadata(document_id="paper-2", chunk_id="paper-2:0"))]


def test_bm25_empty_query_is_safe():
    assert BM25Retriever(documents()).search("") == []


def test_hybrid_deduplicates_and_ranks():
    docs = documents()
    result = HybridRetriever(BM25Retriever(docs), DenseRetriever(docs)).search("uncertainty sensor fusion")
    assert result[0].metadata.document_id == "paper-1"
    assert len({item.metadata.chunk_id for item in result}) == len(result)