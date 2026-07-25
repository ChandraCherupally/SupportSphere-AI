from __future__ import annotations

from typing import Any, List, TypedDict
from src.ai.models import (
    DecisionResult, 
    RetrievedChunk, 
    SourceDocument, 
    SupportResponse
)


# =============================================================================
# SupportState - System & Trace Controls
# =============================================================================

class SupportState(TypedDict, total=False):
    """
    Shared state object flowing through LangGraph nodes.

    Only stores business data exchanged between nodes.
    Temporary objects (Prompt, LLM, Messages, etc.) must never be stored here.
    """

    # ==========================================================================
    # System & Trace Controls
    # ==========================================================================
    runtime_config: Any  # src.runtime_config.runtime.RuntimeConfig
    execution_trace: Any  # src.graph.trace.ExecutionTrace

    # ==========================================================================
    # Ticket Input
    # ==========================================================================
    issue: str
    subject: str
    company: str

    # ==========================================================================
    # LLM #1 Output: Decision Engine
    # ==========================================================================
    decision_result: DecisionResult
    decision_input_tokens: int
    decision_output_tokens: int
    decision_latency_ms: float

    # Convenience shortcuts copied from decision_result for routing
    intent: str
    routing_strategy: str
    retrieval_required: bool
    selected_company: str
    detected_company: str | None
    verified_company: str
    company_match: bool
    company_reference_found: bool
    routing_status: str
    routing_reason: str
    normalized_issue: str
    normalized_subject: str

    # ==========================================================================
    # Retrieval
    # ==========================================================================
    context: list[RetrievedChunk]
    retrieved_context: list[str]
    retrieved_chunks: list[RetrievedChunk]
    sources: list[SourceDocument]
    num_chunks: int
    token_estimate: int
    retrieval_trace: dict
    reranker: str
    search_mode: str
    retrieval_latency_ms: float

    # ==========================================================================
    # LLM #2 Output: Generation Engine
    # ==========================================================================
    response: SupportResponse
    generation_input_tokens: int
    generation_output_tokens: int
    generation_latency_ms: float
    retriever_used: bool
    llm_used: bool
    template_used: bool

    # ==========================================================================
    # Execution Flags & Diagnostics
    # ==========================================================================
    blocked: bool
    regenerated: bool
    warnings: list[str]
