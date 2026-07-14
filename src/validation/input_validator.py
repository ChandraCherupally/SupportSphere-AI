from __future__ import annotations
from typing import Any

def validate_input(state: dict[str, Any]) -> list[str]:
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
