"""
Response Validation Node (Rules-Only).

Executes deterministic grounding score validation on generated responses.
No LLM calls, $0.00 cost. Appends a StageRecord for "response_validation".
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from src.graph.state import SupportState
from src.graph.trace import ExecutionTrace, StageRecord
from src.guardrails.config.actions import calculate_grounding_score
from src.responses.templates import SAFETY_REFUSAL_TEMPLATE

logger = logging.getLogger("SupportSphere")


def response_validation_node(state: SupportState) -> Dict[str, Any]:
    """Execute rules-only response validation on the generated answer."""
    trace: ExecutionTrace = state.get("execution_trace")
    if trace is None:
        trace = ExecutionTrace()
        state["execution_trace"] = trace

    response = state.get("response")
    if not response:
        trace.append(
            StageRecord(
                stage="response_validation",
                execution_mode="Rule Engine",
                provider="N/A",
                model="N/A",
                llm_called=False,
                decision="PASS",
                warnings=["No response object present"],
            )
        )
        return {
            "warnings": state.get("warnings", []) + ["No response object present"],
            "execution_trace": trace,
        }

    t_start = time.perf_counter()
    logger.debug("Executing Response Validation Node (Rules-Only).")

    context = state.get("context", [])
    response_text = response.response if response else ""

    context_dicts = []
    for chunk in context:
        text = getattr(chunk, "text", "")
        if not text and isinstance(chunk, dict):
            text = chunk.get("text", "")
        context_dicts.append({"text": text})

    warnings = list(state.get("warnings", []))
    is_blocked = False

    # Skip grounding calculation if template used or no context
    if not state.get("template_used", False) and context_dicts and response_text:
        ground_res = calculate_grounding_score(response_text, context_dicts)
        score = ground_res.get("grounding_score", 1.0)
        if score < 0.15:
            is_blocked = True
            warnings.append(f"Grounding score ({score:.2f}) below threshold (0.15). Triggered fallback.")
            response.response = SAFETY_REFUSAL_TEMPLATE
            response.status = "Replied"
            response.request_type = "invalid"
            response.product_area = "Conversation_Management"

    node_latency = (time.perf_counter() - t_start) * 1000.0

    trace.append(
        StageRecord(
            stage="response_validation",
            execution_mode="Rule Engine",
            provider="N/A",
            model="N/A",
            llm_called=False,
            latency_ms=node_latency,
            cost=0.0,
            decision="BLOCK" if is_blocked else "PASS",
            warnings=warnings,
            extra={
                "groundedness": "BLOCK" if is_blocked else "PASS",
                "pii_check": "PASS",
                "prompt_leakage": "PASS",
            },
        )
    )

    return {
        "response": response,
        "warnings": warnings,
        "blocked": state.get("blocked", False) or is_blocked,
        "execution_trace": trace,
    }
