from __future__ import annotations
from typing import Any

def validate_retrieval(state: dict[str, Any]) -> list[str]:
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
