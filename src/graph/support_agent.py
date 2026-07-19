"""
Public entry point for the SupportSphere AI support agent.
"""

from __future__ import annotations

import logging
from langgraph.graph.state import CompiledStateGraph

from src.ai.models import AgentResponse, SupportTicket
from src.graph.graph import support_graph
from src.graph.state import SupportState

logger = logging.getLogger(__name__)


class SupportAgent:
    """
    Orchestrates invocation of the LangGraph support workflow.
    """

    def __init__(self, graph: CompiledStateGraph = support_graph) -> None:
        self.graph = graph

    def invoke(self, ticket: SupportTicket) -> AgentResponse:
        """
        Accept a SupportTicket, invoke the LangGraph agent state machine,
        and wrap the result in a clean AgentResponse.
        """
        logger.info("============================== Triage Work flow execution started ==============================")
        logger.info(
            "Triage started for ticket. Subject: '%s', Company: '%s'",
            ticket.subject, ticket.company
        )

        # Validate configuration and API keys before executing
        from src.ai.registry import LLMRegistry
        dec_cfg = LLMRegistry.get_stage_config("decision")
        gen_cfg = LLMRegistry.get_stage_config("generation")
        
        if not LLMRegistry.get_api_key(dec_cfg["provider"]):
            raise ValueError(f"Missing API key for Decision Provider '{dec_cfg['provider']}'. Please configure it in your environment or UI sidebar.")
        if not LLMRegistry.get_api_key(gen_cfg["provider"]):
            raise ValueError(f"Missing API key for Generation Provider '{gen_cfg['provider']}'. Please configure it in your environment or UI sidebar.")

        state: SupportState = {
            "issue": ticket.issue.strip(),
            "subject": ticket.subject.strip(),
            "company": ticket.company.strip(),
            "selected_company": ticket.company.strip(),
            "detected_company": ticket.company.strip(),
            "verified_company": ticket.company.strip(),
            "company_match": True,
            "company_confidence": 1.0,
            "classification_confidence": 1.0,
            "reranker": ticket.reranker,
            "search_mode": ticket.search_mode,
            "context": [],
            "retrieved_context": [],
            "retrieved_chunks": [],
            "sources": [],
            "num_chunks": 0,
            "token_estimate": 0,
            "retrieval_trace": {},
            "warnings": [],
            "retrieval_required": True,
            "normalized_issue": ticket.issue.strip(),
            "normalized_subject": ticket.subject.strip(),
            "routing_reason": "",
            "confidence": 1.0,
            "decision_input_tokens": 0,
            "decision_output_tokens": 0,
            "generation_input_tokens": 0,
            "generation_output_tokens": 0,
        }

        # Invoke the Compiled LangGraph Workflow
        result = self.graph.invoke(state)

        logger.info("Triage workflow execution complete. Assembling AgentResponse.")
        logger.info("============================== Triage Work flow execution completed ==============================")
        return AgentResponse(
            response=result["response"],
            sources=result.get("sources", []),
            retrieved_context=result.get("retrieved_context", []),
            retrieved_chunks=result.get("retrieved_chunks", []),
            num_chunks=result.get("num_chunks", 0),
            token_estimate=result.get("token_estimate", 0),
            retrieval_trace=result.get("retrieval_trace", {}),
            retrieval_required=result.get("retrieval_required", True),
            normalized_issue=result.get("normalized_issue", ""),
            normalized_subject=result.get("normalized_subject", ""),
            routing_reason=result.get("routing_reason", ""),
            confidence=result.get("confidence", 1.0),
            classification_confidence=result.get("classification_confidence", 1.0),
            selected_company=result.get("selected_company", ""),
            detected_company=result.get("detected_company", ""),
            verified_company=result.get("verified_company", ""),
            company_match=result.get("company_match", True),
            company_confidence=result.get("company_confidence", 1.0),
            decision_input_tokens=result.get("decision_input_tokens", 0),
            decision_output_tokens=result.get("decision_output_tokens", 0),
            generation_input_tokens=result.get("generation_input_tokens", 0),
            generation_output_tokens=result.get("generation_output_tokens", 0),
        )
