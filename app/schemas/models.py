from typing import Any

from pydantic import BaseModel, Field


class SourceMetadata(BaseModel):
    document_id: str
    title: str = "Untitled source"
    authors: list[str] = Field(default_factory=list)
    publication_year: int | None = None
    source: str = "unknown"
    url: str | None = None
    page_number: int | None = None
    section: str | None = None
    chunk_id: str


class Evidence(BaseModel):
    text: str
    metadata: SourceMetadata
    score: float = 0.0
    retrieval_method: str = "unknown"


class Citation(BaseModel):
    citation_id: str
    document_id: str
    title: str
    page_number: int | None = None
    section: str | None = None
    url: str | None = None


class VerificationResult(BaseModel):
    supported: bool
    citations_valid: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    evidence_sufficient: bool
    needs_retrieval: bool = False
    reason: str = ""


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str = "default"


class ResearchResponse(BaseModel):
    answer: str
    citations: list[Citation]
    sources: list[SourceMetadata]
    verification: VerificationResult
    latency_ms: float
    model_version: str
    retrieval_version: str


class EvaluationRecord(BaseModel):
    question: str
    expected_sources: list[str] = Field(default_factory=list)
    expected_answer_criteria: list[str] = Field(default_factory=list)
    question_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)