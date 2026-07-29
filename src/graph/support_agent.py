"""
Public entry point for the SupportSphere AI support agent.
"""

from __future__ import annotations

import logging
import time
import uuid

from langgraph.graph.state import CompiledStateGraph

from src.ai.models import AgentResponse, DecisionResult, SupportTicket
from src.graph.graph import support_graph
from src.graph.state import SupportState
from src.graph.trace import ExecutionTrace
from src.runtime_config.runtime import RuntimeConfig

import os 

logger = logging.getLogger("SupportSphere")

class SupportAgent:
    """Orchestrates invocation of the 5-node LangGraph support workflow."""

    def __init__(self, graph: CompiledStateGraph = support_graph) -> None:
        self.graph = graph

    def invoke(self, ticket: SupportTicket) -> AgentResponse:
        """
        Accept a SupportTicket, invoke the LangGraph state machine,
        and return a clean AgentResponse with DecisionResult and ExecutionTrace.
        """
        logger.debug("Triage started for ticket. Subject: '%s', Company: '%s'", ticket.subject, ticket.company)

        # Build RuntimeConfig from current src.config (populated by sidebar)
        import src.config as src_cfg

        top_k_val = getattr(ticket, "top_k", None) or getattr(src_cfg, "FINAL_TOP_K", 5)
        src_cfg.FINAL_TOP_K = top_k_val

        runtime_config = RuntimeConfig(
            decision_provider=getattr(src_cfg, "DECISION_PROVIDER", "google"),
            decision_model=getattr(src_cfg, "DECISION_MODEL", "gemini-2.5-flash-lite"),
            generation_provider=getattr(src_cfg, "GENERATION_PROVIDER", "google"),
            generation_model=getattr(src_cfg, "GENERATION_MODEL", "gemini-2.5-flash"),
            google_api_key=getattr(src_cfg, "GOOGLE_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "") or "",
            openai_api_key=getattr(src_cfg, "OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "") or "",
            groq_api_key=getattr(src_cfg, "GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "") or "",
            top_k=top_k_val,
            search_mode=(ticket.search_mode or getattr(src_cfg, "SEARCH_MODE", "hybrid")).lower(),
            reranker=(ticket.reranker or getattr(src_cfg, "RERANKER", "none")).lower(),
        )

        dec_provider = runtime_config.decision_provider
        gen_provider = runtime_config.generation_provider
        if not runtime_config.get_api_key(dec_provider):
            raise ValueError(
                f"Missing API key for Decision Provider '{dec_provider}'. "
                "Please configure it in your environment or the UI sidebar."
            )
        if not runtime_config.get_api_key(gen_provider):
            raise ValueError(
                f"Missing API key for Generation Provider '{gen_provider}'. "
                "Please configure it in your environment or the UI sidebar."
            )

        # Reset guardrail token accumulator for this ticket
        from src.guardrails.service import GuardrailUsageAccumulator

        GuardrailUsageAccumulator.reset()

        ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")

        state: SupportState = {
            # Configuration
            "runtime_config": runtime_config,
            # Observability
            "execution_trace": ExecutionTrace(),
            # Input
            "issue": ticket.issue.strip(),
            "subject": ticket.subject.strip(),
            "company": ticket.company.strip(),
            # Retrieval defaults
            "reranker": ticket.reranker,
            "search_mode": ticket.search_mode,
            "context": [],
            "retrieved_context": [],
            "retrieved_chunks": [],
            "sources": [],
            "num_chunks": 0,
            "token_estimate": 0,
            "retrieval_trace": {},
            "retrieval_required": True,
            # Token usage
            "decision_input_tokens": 0,
            "decision_output_tokens": 0,
            "generation_input_tokens": 0,
            "generation_output_tokens": 0,
            # Flags & Diagnostics
            "blocked": False,
            "regenerated": False,
            "warnings": [],
        }

        result = self.graph.invoke(state)

        logger.debug("Triage workflow execution complete. Assembling AgentResponse.")

        exec_trace: ExecutionTrace = result.get("execution_trace", ExecutionTrace())
        execution_trace_list = exec_trace.to_list()

        dec_res: DecisionResult = result.get("decision_result")
        if dec_res is None:
            dec_res = DecisionResult(
                safe=not result.get("blocked", False),
                intent=result.get("intent", "PRODUCT_SUPPORT"),
                company=result.get("verified_company", ticket.company),
                routing_strategy=result.get("routing_strategy", "SUPPORT_RESPONSE"),
                retrieval_required=result.get("retrieval_required", True),
                reason=result.get("routing_reason", ""),
                company_reference_found=result.get("company_reference_found", False),
                routing_status=result.get("routing_status", "NO_COMPANY_REFERENCE"),
                normalized_issue=ticket.issue.strip(),
                normalized_subject=ticket.subject.strip(),
            )

        res = AgentResponse(
            response=result["response"],
            decision_result=dec_res,
            sources=result.get("sources", []),
            retrieved_context=result.get("retrieved_context", []),
            retrieved_chunks=result.get("retrieved_chunks", []),
            num_chunks=result.get("num_chunks", 0),
            token_estimate=result.get("token_estimate", 0),
            retrieval_trace=result.get("retrieval_trace", {}),
            retrieval_required=dec_res.retrieval_required,
            normalized_issue=dec_res.normalized_issue,
            normalized_subject=dec_res.normalized_subject,
            routing_reason=dec_res.reason,
            confidence=dec_res.confidence,
            classification_confidence=dec_res.confidence,
            selected_company=ticket.company.strip(),
            detected_company=result.get("detected_company"),
            verified_company=result.get("verified_company", ticket.company),
            company_match=result.get("company_match", True),
            decision_input_tokens=result.get("decision_input_tokens", 0),
            decision_output_tokens=result.get("decision_output_tokens", 0),
            generation_input_tokens=result.get("generation_input_tokens", 0),
            generation_output_tokens=result.get("generation_output_tokens", 0),
            blocked=result.get("blocked", False),
            regenerated=result.get("regenerated", False),
            company_reference_found=dec_res.company_reference_found,
            routing_status=dec_res.routing_status,
            decision_latency_ms=result.get("decision_latency_ms", 0.0),
            generation_latency_ms=result.get("generation_latency_ms", 0.0),
            retrieval_latency_ms=result.get("retrieval_latency_ms", 0.0),
            response_strategy=dec_res.routing_strategy,
            intent=dec_res.intent,
            routing_strategy=dec_res.routing_strategy,
            retriever_used=result.get("retriever_used", False),
            llm_used=result.get("llm_used", False),
            template_used=result.get("template_used", False),
            execution_trace=execution_trace_list,
            warnings=result.get("warnings", []),
        )

        from src.logging import log_ticket_summary

        res_dict = res.model_dump()
        res_dict["ticket_id"] = ticket_id
        res_dict["start_time"] = start_time
        res_dict["subject"] = ticket.subject.strip()
        res_dict["issue"] = ticket.issue.strip()
        res_dict["decision_model"] = runtime_config.decision_model
        res_dict["search_mode"] = ticket.search_mode
        res_dict["reranker"] = ticket.reranker
        res_dict["top_k"] = ticket.top_k
        log_ticket_summary(res_dict)

        return res
