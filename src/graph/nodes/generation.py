"""
Generation Engine Node (LLM #2).

Responsible for generating grounded customer-facing support responses.
Uses static response templates for non-SUPPORT_RESPONSE routing strategies.

Appends a StageRecord for "generation".
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from src.ai.client import LLMClient
from src.ai.models import SupportResponse, SupportTicket
from src.ai.prompt_builder import PromptBuilder
from src.graph.state import SupportState
from src.graph.trace import ExecutionTrace, StageRecord
from src.responses.templates import (
    BUG_ESCALATION_TEMPLATE,
    EMPTY_INPUT_TEMPLATE,
    GREETING_TEMPLATE,
    OUT_OF_SCOPE_TEMPLATE,
    SAFETY_REFUSAL_TEMPLATE,
    THANK_YOU_TEMPLATE,
    UNSUPPORTED_COMPANY_TEMPLATE,
)

logger = logging.getLogger(__name__)

prompt_builder = PromptBuilder()
llm = LLMClient()


def generate_node(state: SupportState) -> Dict[str, Any]:
    """Generate final customer support response via LLM #2 or static template."""
    t_start = time.perf_counter()
    trace: ExecutionTrace = state.get("execution_trace")
    if trace is None:
        trace = ExecutionTrace()
        state["execution_trace"] = trace

    strategy = state.get("routing_strategy", "SUPPORT_RESPONSE")
    intent = state.get("intent", "PRODUCT_SUPPORT")

    # -------------------------------------------------------------------------
    # 1. Template Response Path (Non-SUPPORT_RESPONSE strategies)
    # -------------------------------------------------------------------------
    if strategy != "SUPPORT_RESPONSE":
        logger.debug("Executing Generation Node via static template: %s", strategy)
        reason = state.get("routing_reason", "").lower()

        if strategy in ("SAFETY_REFUSAL_TEMPLATE", "SAFETY_REFUSAL"):
            response_text = (
                EMPTY_INPUT_TEMPLATE
                if (intent == "EMPTY_INPUT" or "empty" in reason)
                else SAFETY_REFUSAL_TEMPLATE
            )
            justification = "Safety policy triggered."
            status = "Replied"
            request_type = "invalid"
        elif strategy == "OUT_OF_SCOPE_TEMPLATE":
            response_text = (
                UNSUPPORTED_COMPANY_TEMPLATE
                if "unsupported company" in reason
                else OUT_OF_SCOPE_TEMPLATE
            )
            justification = "Out-of-scope query."
            status = "Replied"
            request_type = "invalid"
        elif strategy == "GREETING_TEMPLATE":
            response_text = GREETING_TEMPLATE
            justification = "Greeting response."
            status = "Replied"
            request_type = "invalid"
        elif strategy == "THANK_YOU_TEMPLATE":
            response_text = THANK_YOU_TEMPLATE
            justification = "Thank you response."
            status = "Replied"
            request_type = "invalid"
        elif strategy == "BUG_ESCALATION_TEMPLATE":
            response_text = BUG_ESCALATION_TEMPLATE
            justification = "Platform incident escalation."
            status = "Escalated"
            request_type = "bug"
        else:
            response_text = EMPTY_INPUT_TEMPLATE
            justification = "System generated template response."
            status = "Replied"
            request_type = "invalid"

        support_response = SupportResponse(
            request_type=request_type,
            product_area="Conversation_Management",
            status=status,
            response=response_text,
            justification=justification,
        )
        node_latency = (time.perf_counter() - t_start) * 1000.0

        trace.append(
            StageRecord(
                stage="generation",
                execution_mode="Template",
                provider="system",
                model="template",
                llm_called=False,
                latency_ms=node_latency,
                decision="PASS",
            )
        )
        return {
            "response": support_response,
            "generation_input_tokens": 0,
            "generation_output_tokens": 0,
            "generation_latency_ms": node_latency,
            "retriever_used": False,
            "llm_used": False,
            "template_used": True,
            "execution_trace": trace,
        }

    # -------------------------------------------------------------------------
    # 2. LLM #2 Generation Engine Path
    # -------------------------------------------------------------------------
    logger.debug("Executing Generation Node via LLM #2.")
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

    runtime_cfg = state.get("runtime_config")
    if runtime_cfg is not None:
        provider = runtime_cfg.get_stage_provider("generation")
        model_name = runtime_cfg.get_stage_model("generation")
        api_key = runtime_cfg.get_api_key(provider)
    else:
        import src.config as src_config

        provider = src_config.GENERATION_PROVIDER
        model_name = src_config.GENERATION_MODEL
        api_key = src_config.get_api_key(provider)

    from src.billing.catalog import get_model_price

    price = get_model_price(provider.lower(), model_name.lower())

    result = llm.generate(messages=messages, response_schema=SupportResponse, stage="generation", provider=provider, model_name=model_name, api_key=api_key)
    res_obj: SupportResponse = result.response

    if intent == "PRODUCT_SUPPORT":
        res_obj.request_type = "product_issue"
        res_obj.status = "Replied"
    elif intent == "FEATURE_REQUEST":
        res_obj.request_type = "feature_request"
        res_obj.status = "Escalated"
    elif intent == "BUG":
        res_obj.request_type = "bug"
        res_obj.status = "Escalated"
    if not res_obj.product_area or not res_obj.product_area.strip():
        res_obj.product_area = "Conversation_Management"

    cost = 0.0
    if price:
        cost = (result.usage.input_tokens / 1_000_000.0) * price.input_per_million + (
            result.usage.output_tokens / 1_000_000.0
        ) * price.output_per_million

    node_latency = (time.perf_counter() - t_start) * 1000.0
    trace.append(
        StageRecord(
            stage="generation",
            execution_mode="LLM",
            provider=provider,
            model=model_name,
            llm_called=True,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            latency_ms=node_latency,
            cost=cost,
            decision="PASS",
        )
    )

    return {
        "response": res_obj,
        "generation_input_tokens": result.usage.input_tokens,
        "generation_output_tokens": result.usage.output_tokens,
        "generation_latency_ms": node_latency,
        "retriever_used": state.get("retrieval_required", True),
        "llm_used": True,
        "template_used": False,
        "execution_trace": trace,
    }
