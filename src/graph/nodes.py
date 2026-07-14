from __future__ import annotations

from src.graph.state import SupportState

from src.llm.client import LLMClient
from src.llm.models import SupportResponse
from src.llm.prompt_builder import PromptBuilder

from src.retrieval.retriever import Retriever
from src.llm.models import SupportTicket

from src.validation.input_validator import validate_input
from src.validation.retrieval_validator import validate_retrieval
from src.validation.output_validator import validate_and_correct_output

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
    warnings = validate_input(state)
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
    warnings = validate_retrieval(state)
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
    corrected_response, warnings = validate_and_correct_output(state)
    return {
        "response": corrected_response,
        "warnings": state.get("warnings", []) + warnings
    }