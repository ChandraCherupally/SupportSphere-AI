"""
Topological LangGraph workflow wiring for SupportSphere AI.

Pipeline topology:

       START
         │
         ▼
  Decision Engine  (LLM #1: Input Safety + Intent + Company + Strategy + Retrieval Decision)
         │
         ├── [Retrieval Required] ─────► Retrieval ──► Retrieval Validation (Rules)
         │                                                      │
         └── [Retrieval Not Required] ──────────────────────────┤
                                                                │
                                                                ▼
                                                            Generation  (LLM #2 / Template Response)
                                                                │
                                                                ▼
                                                        Response Validation (Rules)
                                                                │
                                                                ▼
                                                               END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    decision_engine_node,
    generate_node,
    response_validation_node,
    retrieval_validation_node,
    retrieve_node,
)
from src.graph.state import SupportState


def build_graph():
    """Build and compile the 5-node stateful LangGraph support agent graph."""
    workflow = StateGraph(SupportState)

    workflow.add_node("decision_engine", decision_engine_node)
    workflow.add_node("retrieval", retrieve_node)
    workflow.add_node("retrieval_validation", retrieval_validation_node)
    workflow.add_node("generation", generate_node)
    workflow.add_node("response_validation", response_validation_node)

    workflow.add_edge(START, "decision_engine")

    def route_after_decision(state: SupportState) -> str:
        if state.get("blocked", False):
            return "generation"
        if state.get("retrieval_required", True):
            return "retrieval"
        return "generation"

    workflow.add_conditional_edges(
        "decision_engine",
        route_after_decision,
        {
            "retrieval": "retrieval",
            "generation": "generation",
        },
    )

    workflow.add_edge("retrieval", "retrieval_validation")
    workflow.add_edge("retrieval_validation", "generation")
    workflow.add_edge("generation", "response_validation")
    workflow.add_edge("response_validation", END)

    return workflow.compile()


support_graph = build_graph()
