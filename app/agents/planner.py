from app.agents.state import ResearchState


def plan_research(state: ResearchState) -> ResearchState:
    question = state["user_question"].strip()
    queries = [question]
    if any(marker in question.lower() for marker in ("compare", "main approaches", "limitations", "evidence")):
        queries.append(f"{question} limitations findings")
    return {"research_plan": {"intent": "research_question", "sources": ["papers", "web"], "multi_search": len(queries) > 1}, "search_queries": queries, "retrieval_attempts": state.get("retrieval_attempts", 0)}