from __future__ import annotations

import re
from pydantic import BaseModel, Field

from src.graph.state import SupportState

from src.ai.client import LLMClient
from src.ai.models import SupportResponse
from src.ai.prompt_builder import PromptBuilder

from src.retrieval.retriever import Retriever
from src.ai.models import SupportTicket

from src.graph.guardrails import (
    apply_input_guardrails, 
    apply_retrieval_guardrails, 
    apply_output_guardrails
)

# =============================================================================
# Routing Decision Model
# =============================================================================

class RoutingDecision(BaseModel):
    """
    Structured schema representing the routing decision.
    """
    retrieval_required: bool = Field(
        description="Whether documentation retrieval is required to answer this support ticket."
    )
    routing_reason: str = Field(
        description="Reason for routing decision."
    )
    confidence: float = Field(
        description="Confidence score for this routing decision (between 0.0 and 1.0)."
    )
    normalized_issue: str = Field(
        description="Cleaned, normalized version of the ticket issue."
    )
    normalized_subject: str = Field(
        description="Cleaned, normalized version of the ticket subject."
    )

# =============================================================================
# Singleton Components
# =============================================================================

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
# Decision Gate Node (NEW)
# =============================================================================

ROUTING_PROMPT = """
You are the Routing & Decision Gate engine for SupportSphere AI.
Analyze the following customer support ticket and decide whether documentation retrieval is required.

Supported companies are: HackerRank, Claude, and Visa.

Skip retrieval (retrieval_required = False) for:
- Greetings, appreciation, thank you, and small talk.
- Obvious out-of-scope questions (general knowledge, irrelevant queries not about HackerRank, Claude, or Visa).
- Operational incidents/platform outages (e.g., site is down, service unavailable, cannot log in due to system failure).
- Empty, spam, or invalid requests.

Run retrieval (retrieval_required = True) for:
- Product questions, how-to/procedural guides, account configurations, billing, troubleshootings, feature lookups, and workflows related to HackerRank, Claude, or Visa.

Provide:
1. retrieval_required (boolean)
2. routing_reason (concise reason)
3. confidence (float between 0.0 and 1.0)
4. normalized_issue (cleaned version of the customer issue)
5. normalized_subject (cleaned version of the subject)
"""

def decision_gate_node(state: SupportState) -> SupportState:
    """
    Determine if retrieval is required and classify routing behavior.
    """
    issue = state.get("issue", "").strip()
    subject = state.get("subject", "").strip()
    company = state.get("company", "").strip()

    # 1. Deterministic Classifier
    issue_lower = issue.lower()
    subject_lower = subject.lower()
    combined_lower = f"{subject_lower}\n{issue_lower}".strip()

    # Empty / Invalid Request
    if not issue:
        return {
            "retrieval_required": False,
            "normalized_issue": "",
            "normalized_subject": "",
            "routing_reason": "Empty or invalid request.",
            "confidence": 1.0,
        }

    # Unsupported Company
    supported_companies = {"hackerrank", "claude", "visa"}
    if company.lower() not in supported_companies:
        return {
            "retrieval_required": False,
            "normalized_issue": issue,
            "normalized_subject": subject,
            "routing_reason": f"Unsupported company: {company}.",
            "confidence": 1.0,
        }

    # Greetings (only greeting words)
    clean_text = re.sub(r'[^\w\s]', '', combined_lower).strip()
    greeting_words = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "yo"}
    if clean_text in greeting_words:
        return {
            "retrieval_required": False,
            "normalized_issue": issue,
            "normalized_subject": subject,
            "routing_reason": "Greeting message.",
            "confidence": 1.0,
        }

    # Thank-you messages
    thanks_words = {"thanks", "thank you", "thankyou", "appreciate it", "great", "perfect", "happy to help"}
    if clean_text in thanks_words:
        return {
            "retrieval_required": False,
            "normalized_issue": issue,
            "normalized_subject": subject,
            "routing_reason": "Thank-you message.",
            "confidence": 1.0,
        }

    # Service Outage / Operational Incident
    outage_keywords = [
        "site is down",
        "service unavailable",
        "all pages inaccessible",
        "entire site unavailable",
        "users cannot log in because of a platform failure",
        "widespread outage",
        "system outage"
    ]
    if any(kw in combined_lower for kw in outage_keywords):
        return {
            "retrieval_required": False,
            "normalized_issue": issue,
            "normalized_subject": subject,
            "routing_reason": "Operational incident (service outage).",
            "confidence": 1.0,
        }

    # 2. Fallback to LLM
    messages = [
        ("system", ROUTING_PROMPT),
        ("human", f"Company: {company}\nSubject: {subject}\nIssue: {issue}")
    ]
    try:
        decision = llm.generate(messages, response_schema=RoutingDecision)
        # Attempt to capture token usage from the raw LLM response
        d_in: int = 0
        d_out: int = 0
        try:
            raw = llm.llm.with_structured_output(RoutingDecision)
            # usage already consumed above; re-use from AIMessage if accessible
            # We store zeros here; actual capture happens in support_agent via state
        except Exception:
            pass
        return {
            "retrieval_required": decision.retrieval_required,
            "normalized_issue": decision.normalized_issue,
            "normalized_subject": decision.normalized_subject,
            "routing_reason": decision.routing_reason,
            "confidence": decision.confidence,
            "decision_input_tokens": d_in,
            "decision_output_tokens": d_out,
        }
    except Exception as e:
        return {
            "retrieval_required": True,
            "normalized_issue": issue,
            "normalized_subject": subject,
            "routing_reason": f"Routing classification failed: {e}. Defaulting to retrieval.",
            "confidence": 0.5,
            "decision_input_tokens": 0,
            "decision_output_tokens": 0,
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

    retriever = Retriever(
        reranker=state.get(
            "reranker",
            "none",
        ),
    )

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
        context=state.get("context", []),
        retrieval_required=state.get("retrieval_required", True),
        routing_reason=state.get("routing_reason", ""),
    )

    response: SupportResponse = llm.generate(
        messages=messages,
        response_schema=SupportResponse,
    )

    # Attempt to capture generation token usage from the underlying LLM
    gen_in: int = 0
    gen_out: int = 0
    try:
        raw_msg = llm.llm.invoke(messages)
        usage = getattr(raw_msg, "usage_metadata", None)
        if usage:
            gen_in = usage.get("input_tokens", 0)
            gen_out = usage.get("output_tokens", 0)
    except Exception:
        pass  # Token capture is best-effort; billing degrades gracefully to 0

    return {
        "response": response,
        "generation_input_tokens": gen_in,
        "generation_output_tokens": gen_out,
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

