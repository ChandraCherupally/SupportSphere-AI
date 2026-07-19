"""
Workflow nodes for SupportSphere AI.
Each node has a single responsibility. Business logic only.
No provider-specific logic or print statements.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Any, List

from pydantic import BaseModel, Field

from src.ai.client import LLMClient
from src.ai.models import SupportResponse, SupportTicket
from src.ai.prompt_builder import PromptBuilder
from src.graph.guardrails import (
    apply_input_guardrails,
    apply_output_guardrails,
    apply_retrieval_guardrails,
)
from src.graph.state import SupportState
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


# =============================================================================
# Routing Decision Model
# =============================================================================

class RoutingDecision(BaseModel):
    """
    Structured schema representing the routing decision.
    """
    selected_company: str = Field(
        description="The company name passed as hint from the user/UI."
    )
    detected_company: str = Field(
        description="The company name inferred from the ticket (one of HackerRank, Claude, Visa, or Unknown)."
    )
    verified_company: str = Field(
        description="The verified company to retrieve documentation for, resolved by applying validation logic."
    )
    company_match: bool = Field(
        description="True if selected_company matches detected_company."
    )
    company_confidence: float = Field(
        description="Confidence score for company detection (between 0.0 and 1.0)."
    )
    retrieval_required: bool = Field(
        description="Whether documentation retrieval is required to answer this support ticket."
    )
    normalized_issue: str = Field(
        description="Cleaned, normalized version of the ticket issue."
    )
    normalized_subject: str = Field(
        description="Cleaned, normalized version of the ticket subject."
    )
    classification_confidence: float = Field(
        description="Confidence score for this routing decision (between 0.0 and 1.0)."
    )
    routing_reason: str = Field(
        description="A single sentence explaining why the company and routing decisions were made (maximum 40 words, no internal reasoning, objective)."
    )


# =============================================================================
# Shared Component Instances
# =============================================================================

prompt_builder = PromptBuilder()
llm = LLMClient()


# =============================================================================
# Input Check Node
# =============================================================================

def input_check_node(state: SupportState) -> Dict[str, Any]:
    """
    Validate ticket input before running LLM classification.
    """
    logger.info("Executing Input Check Node.")
    warnings = apply_input_guardrails(state)
    return {
        "warnings": state.get("warnings", []) + warnings
    }


# =============================================================================
# Decision Gate Node
# =============================================================================

ROUTING_PROMPT = """
You are the Routing & Decision Gate engine for SupportSphere AI.
Analyze the customer support ticket (selected_company hint, subject, issue) and perform company detection, company validation, and routing classification.

1. COMPANY DETECTION:
Infer which company the ticket content actually belongs to (HackerRank, Claude, Visa, or Unknown).
Use subject, issue, and company-specific terminology/product names to determine the correct company.
Examples:
- Claude, Anthropic, Sonnet, Haiku, Opus, bedrock -> Claude
- HackerRank, assessment, candidate, test, question bank, interview, test settings -> HackerRank
- Visa, Traveller Cheque, travelers cheques, Commercial Card, Visa Direct, Merchant, VisaNet -> Visa
Otherwise, if no company-specific terms match, output Unknown.

2. VALIDATION & DECISION LOGIC:
- selected_company: The company passed in the ticket.
- detected_company: The company detected from ticket text (HackerRank, Claude, Visa, or Unknown).
- company_match: true if selected_company matches detected_company, otherwise false.
- company_confidence: Confidence score of company detection (between 0.0 and 1.0).
- verified_company: The resolved company using these confidence thresholds:
  - If company_confidence >= 0.90: verified_company = detected_company.
  - If company_confidence is 0.60 to 0.89: verified_company = detected_company.
  - If company_confidence < 0.60 or detected_company is Unknown: verified_company = selected_company.

3. RETRIEVAL DECISION:
Skip retrieval (retrieval_required = False) for greetings, small talk, general questions, out of scope questions, or system outages.
Otherwise, retrieval_required = True.

4. QUERY NORMALIZATION:
Provide normalized_subject and normalized_issue (cleaned versions without fluff).

5. ROUTING REASON:
Write one sentence (maximum 40 words) explaining why the company was selected/routing was decided.
Must be objective and reference observable evidence from the ticket (e.g. terminology, names). Do NOT use phrases like "I analyzed", "I think", "I inferred", "My reasoning is".
"""


def decision_gate_node(state: SupportState) -> Dict[str, Any]:
    """
    Determine if retrieval is required and classify routing behavior.
    """
    logger.info("Executing Decision Gate Node.")
    
    issue = state.get("issue", "").strip()
    subject = state.get("subject", "").strip()
    company = state.get("company", "").strip()

    # 1. Deterministic Classifiers
    issue_lower = issue.lower()
    subject_lower = subject.lower()
    combined_lower = f"{subject_lower}\n{issue_lower}".strip()

    # Empty / Invalid Request
    if not issue:
        logger.info("Decision Gate: Empty request detected.")
        return {
            "retrieval_required": False,
            "normalized_issue": "",
            "normalized_subject": "",
            "routing_reason": "Empty or invalid request.",
            "confidence": 1.0,
            "classification_confidence": 1.0,
            "selected_company": company,
            "detected_company": "Unknown",
            "verified_company": company,
            "company_match": False,
            "company_confidence": 1.0,
            "decision_input_tokens": 0,
            "decision_output_tokens": 0,
        }

    # Unsupported Company
    supported_companies = {"hackerrank", "claude", "visa"}
    if company.lower() not in supported_companies:
        logger.info("Decision Gate: Unsupported company '%s' detected.", company)
        return {
            "retrieval_required": False,
            "normalized_issue": issue,
            "normalized_subject": subject,
            "routing_reason": f"Unsupported company: {company}.",
            "confidence": 1.0,
            "classification_confidence": 1.0,
            "selected_company": company,
            "detected_company": "Unknown",
            "verified_company": company,
            "company_match": False,
            "company_confidence": 1.0,
            "decision_input_tokens": 0,
            "decision_output_tokens": 0,
        }

    # Greetings
    clean_text = re.sub(r'[^\w\s]', '', combined_lower).strip()
    greeting_words = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "yo"}
    if clean_text in greeting_words:
        logger.info("Decision Gate: Greeting message detected.")
        return {
            "retrieval_required": False,
            "normalized_issue": issue,
            "normalized_subject": subject,
            "routing_reason": "Greeting message.",
            "confidence": 1.0,
            "classification_confidence": 1.0,
            "selected_company": company,
            "detected_company": company,
            "verified_company": company,
            "company_match": True,
            "company_confidence": 1.0,
            "decision_input_tokens": 0,
            "decision_output_tokens": 0,
        }

    # Thank-you messages
    thanks_words = {"thanks", "thank you", "thankyou", "appreciate it", "great", "perfect", "happy to help"}
    if clean_text in thanks_words:
        logger.info("Decision Gate: Thank-you message detected.")
        return {
            "retrieval_required": False,
            "normalized_issue": issue,
            "normalized_subject": subject,
            "routing_reason": "Thank-you message.",
            "confidence": 1.0,
            "classification_confidence": 1.0,
            "selected_company": company,
            "detected_company": company,
            "verified_company": company,
            "company_match": True,
            "company_confidence": 1.0,
            "decision_input_tokens": 0,
            "decision_output_tokens": 0,
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
        logger.info("Decision Gate: Platform outage incident detected.")
        return {
            "retrieval_required": False,
            "normalized_issue": issue,
            "normalized_subject": subject,
            "routing_reason": "Operational incident (service outage).",
            "confidence": 1.0,
            "classification_confidence": 1.0,
            "selected_company": company,
            "detected_company": company,
            "verified_company": company,
            "company_match": True,
            "company_confidence": 1.0,
            "decision_input_tokens": 0,
            "decision_output_tokens": 0,
        }

    # 2. Fallback to LLM Stage: decision
    messages = [
        ("system", ROUTING_PROMPT),
        ("human", f"Selected Company Hint: {company}\nSubject: {subject}\nIssue: {issue}")
    ]
    
    try:
        result = llm.generate(messages, response_schema=RoutingDecision, stage="decision")
        decision = result.response
        
        logger.info("Decision Gate LLM classification successful: %s", decision)
        
        # Enforce validation logic thresholds deterministically in Python
        detected = str(decision.detected_company).strip()
        conf = float(decision.company_confidence)
        
        supported_companies = {"hackerrank", "claude", "visa"}
        detected_normalized = next((c for c in supported_companies if c in detected.lower()), "unknown")
        
        if detected_normalized != "unknown":
            detected_standard = "HackerRank" if detected_normalized == "hackerrank" else ("Claude" if detected_normalized == "claude" else "Visa")
            if conf >= 0.60:
                verified_company = detected_standard
            else:
                verified_company = company
        else:
            verified_company = company
            
        company_match = (company.lower() == detected_normalized)

        # Logging details before retrieval begins
        logger.info(
            "Selected Company: %s\n"
            "Detected Company: %s\n"
            "Verified Company: %s\n"
            "Company Match: %s\n"
            "Company Confidence: %.1f%%\n"
            "Routing Reason: %s",
            company, detected, verified_company, company_match, conf * 100, decision.routing_reason
        )
        
        return {
            "retrieval_required": decision.retrieval_required,
            "normalized_issue": decision.normalized_issue,
            "normalized_subject": decision.normalized_subject,
            "routing_reason": decision.routing_reason,
            "confidence": decision.classification_confidence,
            "classification_confidence": decision.classification_confidence,
            "selected_company": company,
            "detected_company": detected,
            "verified_company": verified_company,
            "company_match": company_match,
            "company_confidence": conf,
            "decision_input_tokens": result.usage.input_tokens,
            "decision_output_tokens": result.usage.output_tokens,
        }
    except Exception as e:
        logger.error("Decision Gate LLM execution failed: %s. Defaulting to retrieval.", e, exc_info=True)
        return {
            "retrieval_required": True,
            "normalized_issue": issue,
            "normalized_subject": subject,
            "routing_reason": f"Routing classification failed: {e}. Defaulting to retrieval.",
            "confidence": 0.5,
            "classification_confidence": 0.5,
            "selected_company": company,
            "detected_company": company,
            "verified_company": company,
            "company_match": True,
            "company_confidence": 0.5,
            "decision_input_tokens": 0,
            "decision_output_tokens": 0,
        }


# =============================================================================
# Retrieve Node
# =============================================================================

def retrieve_node(state: SupportState) -> Dict[str, Any]:
    """
    Retrieve the most relevant documentation chunks.
    """
    logger.info("Executing Retrieve Node.")
    
    retriever = Retriever(
        reranker=state.get("reranker", "none"),
        search_mode=state.get("search_mode", "hybrid"),
    )

    verified_co = state.get("verified_company", state.get("company", ""))
    retrieval = retriever.retrieve(
        issue=state["issue"],
        subject=state.get("subject", ""),
        company=verified_co,
    )

    retrieval["retrieved_context"] = [
        chunk["text"]
        for chunk in retrieval["context"]
    ]

    # Populate trace details requested by the user
    import src.config as src_config
    trace = retrieval.get("retrieval_trace", {})
    trace["selected_company"] = state.get("selected_company", state.get("company", ""))
    trace["detected_company"] = state.get("detected_company", "")
    trace["verified_company"] = verified_co
    trace["collection_used"] = "support-sphere-ai"
    trace["top_k"] = getattr(src_config, "FINAL_TOP_K", 10)
    trace["routing_reason"] = state.get("routing_reason", "")

    # If LLM Reranker was used, accumulate its token usage into Decision totals
    from src.retrieval.rerankers.llm_reranker import LLMReranker
    reranker_instance = retriever.hybrid.reranker
    if isinstance(reranker_instance, LLMReranker):
        dec_in = state.get("decision_input_tokens", 0) + reranker_instance.input_tokens
        dec_out = state.get("decision_output_tokens", 0) + reranker_instance.output_tokens
        retrieval["decision_input_tokens"] = dec_in
        retrieval["decision_output_tokens"] = dec_out
        
        # Reset instance tokens after accumulation
        reranker_instance.input_tokens = 0
        reranker_instance.output_tokens = 0

    return retrieval


# =============================================================================
# Retrieval Check Node
# =============================================================================

def retrieval_check_node(state: SupportState) -> Dict[str, Any]:
    """
    Validate retrieved documentation context via guardrails.
    """
    logger.info("Executing Retrieval Check Node.")
    warnings = apply_retrieval_guardrails(state)
    return {
        "warnings": state.get("warnings", []) + warnings
    }


# =============================================================================
# Generate Node
# =============================================================================

def generate_node(state: SupportState) -> Dict[str, Any]:
    """
    Generate the final support response using retrieved documentation.
    """
    logger.info("Executing Generate Node.")
    
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

    result = llm.generate(
        messages=messages,
        response_schema=SupportResponse,
        stage="generation",
    )

    logger.info("Generation successful. Tokens consumed: Input=%d, Output=%d", 
                result.usage.input_tokens, result.usage.output_tokens)

    return {
        "response": result.response,
        "generation_input_tokens": result.usage.input_tokens,
        "generation_output_tokens": result.usage.output_tokens,
    }


# =============================================================================
# Output Check Node
# =============================================================================

def output_check_node(state: SupportState) -> Dict[str, Any]:
    """
    Validate and correct the generated response.
    """
    logger.info("Executing Output Check Node.")
    corrected_response, warnings = apply_output_guardrails(state)
    return {
        "response": corrected_response,
        "warnings": state.get("warnings", []) + warnings
    }
