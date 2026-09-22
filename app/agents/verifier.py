import re

from app.agents.state import ResearchState
from app.schemas.models import VerificationResult


def verify(state: ResearchState) -> ResearchState:
    evidence_ids = {f"S{index}" for index, _ in enumerate(state.get("retrieved_evidence", []), 1)}
    cited_ids = set(re.findall(r"\[(S\d+)\]", state.get("draft_answer", "")))
    valid = cited_ids.issubset(evidence_ids)
    sufficient = bool(state.get("retrieved_evidence"))
    citations_present_when_needed = not sufficient or bool(cited_ids)
    result = VerificationResult(supported=valid and sufficient, citations_valid=valid and citations_present_when_needed, evidence_sufficient=sufficient, needs_retrieval=not sufficient and state.get("retrieval_attempts", 0) < 2, reason="Citations must reference retrieved evidence.")
    return {"verification_result": result}