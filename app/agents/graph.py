from app.agents.planner import plan_research
from app.agents.researcher import research
from app.agents.state import ResearchState
from app.agents.synthesizer import synthesize
from app.agents.verifier import verify
from app.llm.llama import LLMProvider
from app.retrieval.hybrid import HybridRetriever


class ResearchGraph:
    """Explicit workflow facade; uses LangGraph when available and a transparent runner otherwise."""

    def __init__(self, retriever: HybridRetriever, llm: LLMProvider, top_k: int = 5) -> None:
        self.retriever, self.llm, self.top_k = retriever, llm, top_k
        self._compiled = self._build_langgraph()

    def _build_langgraph(self):
        try:
            from langgraph.graph import END, START, StateGraph

            workflow = StateGraph(ResearchState)
            workflow.add_node("planner", plan_research)
            workflow.add_node("researcher", lambda state: research(state, self.retriever, self.top_k))
            workflow.add_node("synthesizer", lambda state: synthesize(state, self.llm))
            workflow.add_node("verifier", verify)
            workflow.add_edge(START, "planner")
            workflow.add_edge("planner", "researcher")
            workflow.add_edge("researcher", "synthesizer")
            workflow.add_edge("synthesizer", "verifier")
            workflow.add_edge("verifier", END)
            return workflow.compile()
        except (ImportError, RuntimeError, TypeError, ValueError):
            return None

    def invoke(self, question: str, session_id: str = "default") -> ResearchState:
        state: ResearchState = {"user_question": question, "session_id": session_id, "errors": [], "model_version": "llama3", "prompt_version": "v1", "retrieval_version": "hybrid-v1"}
        if self._compiled:
            return self._compiled.invoke(state)
        state.update(plan_research(state))
        state.update(research(state, self.retriever, self.top_k))
        state.update(synthesize(state, self.llm))
        state.update(verify(state))
        return state