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
            "reranker": ticket.reranker,
            # Initialize empty defaults for safety
            "context": [],
            "retrieved_context": [],
            "retrieved_chunks": [],
            "sources": [],
            "num_chunks": 0,
            "token_estimate": 0,
            "warnings": [],
            "retrieval_required": True,
            "normalized_issue": ticket.issue.strip(),
            "normalized_subject": ticket.subject.strip(),
            "routing_reason": "",
            "confidence": 1.0,
        }

        result = self.graph.invoke(state)

        return AgentResponse(
            response=result["response"],
            sources=result.get("sources", []),
            retrieved_context=result.get("retrieved_context", []),
            retrieved_chunks=result.get("retrieved_chunks", []),
            num_chunks=result.get("num_chunks", 0),
            token_estimate=result.get("token_estimate", 0),
            retrieval_required=result.get("retrieval_required", True),
            normalized_issue=result.get("normalized_issue", ""),
            normalized_subject=result.get("normalized_subject", ""),
            routing_reason=result.get("routing_reason", ""),
            confidence=result.get("confidence", 1.0),
        )
