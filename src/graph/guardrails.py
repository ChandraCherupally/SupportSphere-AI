from __future__ import annotations
from typing import Any


def apply_input_guardrails(state: dict[str, Any]) -> list[str]:
    """
    Validate ticket inputs before retrieval.
    Returns a list of warning/error messages.
    """
    warnings = []
    
    company = state.get("company", "").strip().lower()
    supported_companies = {"hackerrank", "claude", "visa"}
    
    if company not in supported_companies:
        warnings.append(f"Unsupported company: '{state.get('company')}'")
        
    issue = state.get("issue", "").strip()
    if not issue:
        warnings.append("Ticket issue is empty")
        
    return warnings



def apply_retrieval_guardrails(state: dict[str, Any]) -> list[str]:
    """
    Validate the retrieved context.
    Returns a list of warning/error messages.
    """
    warnings = []
    context = state.get("context", [])
    
    if not context:
        warnings.append("No context chunks retrieved")
    else:
        ticket_company = state.get("company", "").strip().lower()
        for i, chunk in enumerate(context):
            # context chunks can be RetrievedChunk models or dicts depending on where they come from
            chunk_company = getattr(chunk, "company", "")
            if not chunk_company and isinstance(chunk, dict):
                chunk_company = chunk.get("company", "")
                
            if chunk_company.strip().lower() != ticket_company:
                warnings.append(
                    f"Company mismatch in chunk {i+1}: expected '{state.get('company')}', got '{chunk_company}'"
                )
                
    return warnings




def apply_output_guardrails(state: dict[str, Any]) -> tuple[Any, list[str]]:
    """
    Validate and programmatically correct the output response.
    Returns (corrected_response, warnings).
    """
    warnings = []
    response = state.get("response")
    if not response:
        warnings.append("No response generated")
        return response, warnings
        
    issue_lower = state.get("issue", "").lower()
    
    # 1. Live outage detection guardrail
    if "site is down" in issue_lower or "none of the pages are accessible" in issue_lower:
        if (response.request_type != "bug" or 
            response.status != "Escalated" or 
            response.response != "Escalate to a human" or 
            response.product_area != ""):
            
            warnings.append("Outage guardrail triggered: corrected response fields.")
            response.request_type = "bug"
            response.status = "Escalated"
            response.response = "Escalate to a human"
            response.product_area = ""
            
    # 2. General Escalated status consistency check
    elif response.status == "Escalated":
        if response.product_area != "":
            warnings.append("Consistency warning: cleared product_area for escalated status.")
            response.product_area = ""
            
    # 3. Invalid request type check (Greetings vs. Out of scope)
    elif response.request_type == "invalid":
        if "thank you" in issue_lower or "thanks" in issue_lower or issue_lower.strip() == "happy to help":
            if response.response != "Happy to help" or response.product_area != "":
                warnings.append("Greetings guardrail triggered: corrected response to 'Happy to help'.")
                response.product_area = ""
                response.response = "Happy to help"
        else:
            if (response.response != "I am sorry, this is out of scope from my capabilities" or 
                response.product_area != "conversation_management"):
                
                warnings.append("Out-of-scope guardrail triggered: corrected response.")
                response.product_area = "conversation_management"
                response.response = "I am sorry, this is out of scope from my capabilities"
                
    return response, warnings


