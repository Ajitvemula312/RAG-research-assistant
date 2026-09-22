from app.schemas.models import ResearchRequest, SourceMetadata


def test_request_and_metadata_are_structured():
    request = ResearchRequest(question="What is supported?", session_id="s1")
    metadata = SourceMetadata(document_id="d", chunk_id="d:0")
    assert request.session_id == "s1"
    assert metadata.source == "unknown"