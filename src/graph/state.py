from __future__ import annotations

from typing import TypedDict

from src.ai.models import RetrievedChunk
from src.ai.models import SupportResponse
from src.ai.models import SourceDocument


class SupportState(TypedDict, total=False):
    """
    Shared LangGraph state.

    Only stores business data exchanged between nodes.
    Temporary objects (Prompt, LLM, Messages, etc.)
    should never be stored in the graph state.
    """

    # =======================================================================
    # Input
    # =======================================================================

    issue: str

    subject: str

    company: str

    # =======================================================================
    # Decision Gate (NEW)
    # =======================================================================

    retrieval_required: bool

    normalized_issue: str

    normalized_subject: str

    routing_reason: str

    confidence: float

    # ===========================================================================
    # Retrieval
    # ===========================================================================

    context: list[RetrievedChunk]

    # Plain text context used for evaluation / observability
    retrieved_context: list[str]

    retrieved_chunks: list[RetrievedChunk]

    sources: list[SourceDocument]

    num_chunks: int

    token_estimate: int

    # ===========================================================================
    # Validation
    # ===========================================================================

    warnings: list[str]
    # ===========================================================================
    # reranker
    # ===========================================================================

    reranker: str

    # =======================================================================
    # Output
    # =======================================================================

    response: SupportResponse

    # =======================================================================
    # Token Usage (for Billing Engine)
    # =======================================================================

    decision_input_tokens: int
    decision_output_tokens: int
    generation_input_tokens: int
    generation_output_tokens: int