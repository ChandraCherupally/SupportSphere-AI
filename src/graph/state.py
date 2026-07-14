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

    # ===========================================================================
    # Retrieval
    # ===========================================================================

    context: list[RetrievedChunk]

    # Plain text context used for evaluation / observability
    retrieved_context: list[str]

    sources: list[SourceDocument]

    num_chunks: int

    token_estimate: int

    # ===========================================================================
    # Validation
    # ===========================================================================

    warnings: list[str]
    # =======================================================================
    # Output
    # =======================================================================

    response: SupportResponse 