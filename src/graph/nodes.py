from __future__ import annotations

from src.graph.state import SupportState

from src.ai.client import LLMClient
from src.ai.models import SupportResponse
from src.ai.prompt_builder import PromptBuilder

from src.retrieval.retriever import Retriever
from src.ai.models import SupportTicket

from graph.guardrails import (
    apply_input_guardrails, 
    apply_retrieval_guardrails, 
    apply_output_guardrails
)

# =============================================================================
# Singleton Components
# =============================================================================

retriever = Retriever()

prompt_builder = PromptBuilder()

llm = LLMClient()


# =============================================================================
# Input Check Node
# =============================================================================

def input_check_node(state: SupportState) -> SupportState:
    """
    Validate ticket input before retrieval.
    """
    warnings = apply_input_guardrails(state)
    return {
        "warnings": state.get("warnings", []) + warnings
    }


# =============================================================================
# Retrieve Node
# =============================================================================

def retrieve_node(state: SupportState) -> SupportState:
    """
    Retrieve the most relevant documentation.

    Output:
        context
        sources
        num_chunks
        token_estimate
    """

    retrieval = retriever.retrieve(
        issue=state["issue"],
        subject=state.get("subject", ""),
        company=state.get("company", ""),
    )

    retrieval["retrieved_context"] = [
        chunk["text"]
        for chunk in retrieval["context"]
    ]

    return retrieval


# =============================================================================
# Retrieval Check Node
# =============================================================================

def retrieval_check_node(state: SupportState) -> SupportState:
    """
    Validate retrieved documentation context.
    """
    warnings = apply_retrieval_guardrails(state)
    return {
        "warnings": state.get("warnings", []) + warnings
    }


# =============================================================================
# Generate Node
# =============================================================================

def generate_node(state: SupportState) -> SupportState:
    """
    Generate the final structured response.
    """
    ticket = SupportTicket(
        issue=state["issue"],
        subject=state.get("subject", ""),
        company=state.get("company", ""),
    )
    
    messages = prompt_builder.build(
        ticket=ticket,
        context=state["context"],
    )

    response: SupportResponse = llm.generate(
        messages=messages,
        response_schema=SupportResponse,
    )

    return {
        "response": response,
    }


# =============================================================================
# Output Check Node
# =============================================================================

def output_check_node(state: SupportState) -> SupportState:
    """
    Validate and correct the generated response.
    """
    corrected_response, warnings = apply_output_guardrails(state)
    return {
        "response": corrected_response,
        "warnings": state.get("warnings", []) + warnings
    }

