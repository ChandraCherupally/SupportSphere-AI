from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph

from src.graph.graph import support_graph
from src.graph.state import SupportState
from src.ai.models import (AgentResponse, SupportTicket,)


class SupportAgent:
    """
    Public entry point for SupportSphere AI.

    Responsibilities
    ----------------
    • Accept support ticket input
    • Execute the LangGraph workflow
    • Return a structured response

    The agent intentionally knows nothing about:

    - Pinecone
    - BM25
    - PromptBuilder
    - LangChain
    - LLM Providers
    """

    def __init__(self, graph: CompiledStateGraph = support_graph,) -> None:
        self.graph = graph

    def invoke(self, ticket: SupportTicket,) -> AgentResponse:

        state: SupportState = {
            "issue": ticket.issue.strip(),
            "subject": ticket.subject.strip(),
            "company": ticket.company.strip(),
        }

        result = self.graph.invoke(state)
        
        return AgentResponse(
            response=result["response"],
            sources=result.get("sources", []),
            retrieved_context=result.get("retrieved_context",[],),
            num_chunks=result.get("num_chunks", 0),
            token_estimate=result.get("token_estimate", 0),
        )


