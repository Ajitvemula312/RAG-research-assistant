from typing import TypedDict

from app.schemas.models import Citation, Evidence, VerificationResult


class ResearchState(TypedDict, total=False):
    session_id: str
    user_question: str
    research_plan: dict
    search_queries: list[str]
    retrieved_documents: list[Evidence]
    retrieved_evidence: list[Evidence]
    draft_answer: str
    citations: list[Citation]
    verification_result: VerificationResult
    errors: list[str]
    model_version: str
    prompt_version: str
    retrieval_version: str
    retrieval_attempts: int