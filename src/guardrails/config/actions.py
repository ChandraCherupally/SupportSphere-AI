import logging
from typing import Any, Dict

from nemoguardrails.actions.actions import action

logger = logging.getLogger(__name__)


def get_field(chunk: Any, field_name: str, default: Any = "") -> Any:
    """Safely retrieves a field value from a chunk object or dict."""
    if chunk is None:
        return default
    if isinstance(chunk, str):
        if field_name == "text":
            return chunk
        return default
    if isinstance(chunk, dict):
        return chunk.get(field_name, default)
    # Check for object attributes
    val = getattr(chunk, field_name, None)
    if val is not None:
        return val
    # Fallback to key-access if __getitem__ is available
    try:
        return chunk[field_name]
    except Exception:
        return default


@action(name="verify_company")
def verify_company(company: Any, context: Any) -> Dict[str, Any]:
    """Verify if retrieved chunks belong to the correct company."""
    logger.info("Executing custom action: verify_company (company=%s)", company)

    # Defensive check for unresolved Colang variables or empty fields
    if (
        not company
        or not isinstance(company, str)
        or company.startswith("$")
        or company.lower() == "none"
    ):
        return {"decision": "ALLOW", "reason": "No valid company provided."}

    if (
        not context
        or not isinstance(context, list)
        or (isinstance(context, str) and context.startswith("$"))
    ):
        return {"decision": "ALLOW", "reason": "No valid context provided."}

    mismatches = 0
    ticket_company = company.strip().lower()
    for idx, chunk in enumerate(context):
        chunk_company = get_field(chunk, "company", "").strip().lower()
        if chunk_company and chunk_company != ticket_company:
            mismatches += 1

    if mismatches > 0:
        return {
            "decision": "BLOCK",
            "reason": f"Company mismatch detected in {mismatches} chunks.",
            "rule": "verify_company",
        }
    return {
        "decision": "ALLOW",
        "reason": "All chunks match company.",
        "rule": "verify_company",
    }


@action(name="filter_duplicate_chunks")
def filter_duplicate_chunks(context: Any) -> Dict[str, Any]:
    """Remove duplicate text chunks to optimize prompt context."""
    logger.info("Executing custom action: filter_duplicate_chunks")

    if (
        not context
        or not isinstance(context, list)
        or (isinstance(context, str) and context.startswith("$"))
    ):
        return {"filtered_context_indices": []}

    seen_texts = set()
    filtered_context_indices = []

    for idx, chunk in enumerate(context):
        text = get_field(chunk, "text", "").strip()
        if text and text not in seen_texts:
            seen_texts.add(text)
            filtered_context_indices.append(idx)

    return {"filtered_context_indices": filtered_context_indices}


@action(name="calculate_grounding_score")
def calculate_grounding_score(response: Any, context: Any) -> Dict[str, Any]:
    """Calculate grounding score based on textual overlap."""
    logger.info("Executing custom action: calculate_grounding_score")

    if not response or not isinstance(response, str) or response.startswith("$"):
        return {"grounding_score": 1.0}

    if (
        not context
        or not isinstance(context, list)
        or (isinstance(context, str) and context.startswith("$"))
    ):
        return {"grounding_score": 1.0}

    response_words = set(response.lower().split())
    context_words = set()
    for chunk in context:
        text_val = get_field(chunk, "text", "")
        if text_val:
            context_words.update(text_val.lower().split())

    if not response_words:
        return {"grounding_score": 1.0}

    overlap = len(response_words.intersection(context_words))
    score = overlap / len(response_words)
    return {"grounding_score": score}
