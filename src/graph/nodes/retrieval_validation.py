"""
Retrieval Validation Node (Rules-Only).

Executes deterministic validation checks on retrieved chunks:
- Company match verification
- Duplicate chunk filtering

No LLM calls, $0.00 cost. Appends a StageRecord for "retrieval_validation".
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from src.graph.state import SupportState
from src.graph.trace import ExecutionTrace, StageRecord
from src.guardrails.config.actions import filter_duplicate_chunks, verify_company

logger = logging.getLogger(__name__)


def retrieval_validation_node(state: SupportState) -> Dict[str, Any]:
    """Execute rules-only retrieval context validation."""
    trace: ExecutionTrace = state.get("execution_trace")
    if trace is None:
        trace = ExecutionTrace()
        state["execution_trace"] = trace

    context = state.get("context", [])
    if not state.get("retrieval_required", True) or state.get("blocked", False) or not context:
        trace.append(
            StageRecord(
                stage="retrieval_validation",
                execution_mode="Skipped",
                provider="N/A",
                model="N/A",
                llm_called=False,
                decision="PASS",
            )
        )
        return {
            "warnings": state.get("warnings", []),
            "context": [],
            "execution_trace": trace,
        }

    t_start = time.perf_counter()
    logger.debug("Executing Retrieval Validation Node (Rules-Only).")

    company = state.get("verified_company", state.get("company", ""))
    context_dicts = []
    for chunk in context:
        text = getattr(chunk, "text", "")
        if not text and isinstance(chunk, dict):
            text = chunk.get("text", "")
        chunk_co = getattr(chunk, "company", "")
        if not chunk_co and isinstance(chunk, dict):
            chunk_co = chunk.get("company", "")
        context_dicts.append({"text": text, "company": chunk_co})

    # 1. Company verification rule
    ver_res = verify_company(company, context_dicts)
    is_blocked = ver_res.get("decision") == "BLOCK"
    warnings = list(state.get("warnings", []))
    if is_blocked:
        warnings.append(f"Retrieval Validation Blocked: {ver_res.get('reason')}")

    # 2. Duplicate chunk filtering rule
    dup_res = filter_duplicate_chunks(context_dicts)
    filtered_indices = dup_res.get("filtered_context_indices", list(range(len(context))))
    filtered_context = [context[i] for i in filtered_indices if 0 <= i < len(context)]

    node_latency = (time.perf_counter() - t_start) * 1000.0

    trace.append(
        StageRecord(
            stage="retrieval_validation",
            execution_mode="Rule Engine",
            provider="N/A",
            model="N/A",
            llm_called=False,
            latency_ms=node_latency,
            cost=0.0,
            decision="BLOCK" if is_blocked else "ALLOW",
            warnings=warnings,
            extra={
                "company_verification": "BLOCK" if is_blocked else "PASS",
                "duplicate_filter": "PASS",
            },
        )
    )

    return {
        "context": filtered_context,
        "blocked": state.get("blocked", False) or is_blocked,
        "warnings": warnings,
        "execution_trace": trace,
    }
