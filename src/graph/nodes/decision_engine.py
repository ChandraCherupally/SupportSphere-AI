"""
Decision Engine Node (LLM #1).

Unified reasoning engine responsible for:
- Input Safety Rails (NeMo Guardrails)
- Intent Classification
- Company Detection & Verification
- Routing Strategy Selection
- Retrieval Decision (retrieval_required)

Returns a single DecisionResult object and appends a StageRecord for "decision_engine".
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, Optional

from src.ai.client import LLMClient
from src.ai.models import DecisionResult
from src.ai.prompts import DECISION_ENGINE_PROMPT
from src.graph.state import SupportState
from src.graph.trace import ExecutionTrace, StageRecord
from src.guardrails.service import NeMoGuardrailsService

logger = logging.getLogger("SupportSphere")

llm = LLMClient()


def get_strategy_for_intent(intent: str) -> str:
    """Map intent string to template or support routing strategy."""
    intent_upper = intent.upper()
    if intent_upper == "GREETING":
        return "GREETING_TEMPLATE"
    if intent_upper == "THANK_YOU":
        return "THANK_YOU_TEMPLATE"
    if intent_upper in ("PLATFORM_INCIDENT", "BUG"):
        return "BUG_ESCALATION_TEMPLATE"
    if intent_upper == "OUT_OF_SCOPE":
        return "OUT_OF_SCOPE_TEMPLATE"
    if intent_upper in ("SAFETY_REFUSAL", "EMPTY_INPUT"):
        return "SAFETY_REFUSAL_TEMPLATE"
    return "SUPPORT_RESPONSE"


def decision_engine_node(state: SupportState) -> Dict[str, Any]:
    """
    Single LangGraph node for Decision Engine (LLM #1).

    Executes NeMo safety check first, then performs intent & company routing.
    """
    t_start = time.perf_counter()
    logger.debug("Executing Decision Engine Node.")

    trace: ExecutionTrace = state.get("execution_trace")
    if trace is None:
        trace = ExecutionTrace()
        state["execution_trace"] = trace

    runtime_cfg = state.get("runtime_config")
    if runtime_cfg is not None:
        provider = runtime_cfg.get_stage_provider("decision")
        model_name = runtime_cfg.get_stage_model("decision")
        api_key = runtime_cfg.get_api_key(provider)
    else:
        import src.config as src_config

        provider = src_config.DECISION_PROVIDER
        model_name = src_config.DECISION_MODEL
        api_key = src_config.get_api_key(provider)

    issue = state.get("issue", "").strip()
    subject = state.get("subject", "").strip()
    company = state.get("company", "").strip()

    issue_lower = issue.lower()
    subject_lower = subject.lower()
    combined_lower = f"{subject_lower}\n{issue_lower}".strip()
    clean_text = re.sub(r"[^\w\s]", "", combined_lower).strip()
    clean_words = clean_text.split()

    # -------------------------------------------------------------------------
    # Helper to assemble final node output
    # -------------------------------------------------------------------------
    def _finish(
        dec_res: DecisionResult,
        in_tok: int = 0,
        out_tok: int = 0,
        cost: float = 0.0,
        llm_called: bool = False,
        mode: str = "Deterministic",
    ) -> Dict[str, Any]:
        node_latency = (time.perf_counter() - t_start) * 1000.0

        trace.append(
            StageRecord(
                stage="decision_engine",
                execution_mode="LLM" if llm_called else mode,
                provider=provider if llm_called else "N/A",
                model=model_name if llm_called else "N/A",
                llm_called=llm_called,
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_ms=node_latency,
                cost=cost,
                decision="PASS" if dec_res.safe else "BLOCK",
                extra={"intent": dec_res.intent, "strategy": dec_res.routing_strategy},
            )
        )

        return {
            "decision_result": dec_res,
            "decision_input_tokens": in_tok,
            "decision_output_tokens": out_tok,
            "decision_latency_ms": node_latency,
            "intent": dec_res.intent,
            "routing_strategy": dec_res.routing_strategy,
            "retrieval_required": dec_res.retrieval_required,
            "selected_company": company,
            "detected_company": dec_res.company if dec_res.company_reference_found else None,
            "verified_company": dec_res.company or company,
            "company_match": dec_res.company == company if dec_res.company_reference_found else True,
            "company_reference_found": dec_res.company_reference_found,
            "routing_status": dec_res.routing_status,
            "routing_reason": dec_res.reason,
            "normalized_issue": dec_res.normalized_issue,
            "normalized_subject": dec_res.normalized_subject,
            "blocked": not dec_res.safe,
            "execution_trace": trace,
        }

    # -------------------------------------------------------------------------
    # 1. NeMo Safety Guardrails Check
    # -------------------------------------------------------------------------
    nemo_service = NeMoGuardrailsService()
    if nemo_service.is_available() and issue:
        try:
            nemo_res = nemo_service.check_input(subject, issue)
            if nemo_res.get("decision") == "BLOCK":
                logger.info("Decision Engine: NeMo Safety Guardrail BLOCK.")
                res = DecisionResult(
                    safe=False,
                    intent="SAFETY_REFUSAL",
                    company=company,
                    routing_strategy="SAFETY_REFUSAL_TEMPLATE",
                    retrieval_required=False,
                    reason="Input safety violation detected by safety rails.",
                    normalized_issue=issue,
                    normalized_subject=subject,
                )
                return _finish(res, mode="Safety Engine")
        except Exception as e:
            logger.warning("NeMo check_input error: %s", e)

    # -------------------------------------------------------------------------
    # 2. Deterministic Short-circuits
    # -------------------------------------------------------------------------
    if not issue:
        res = DecisionResult(
            safe=True,
            intent="EMPTY_INPUT",
            company=company,
            routing_strategy="SAFETY_REFUSAL_TEMPLATE",
            retrieval_required=False,
            reason="Empty or invalid request.",
            normalized_issue=issue,
            normalized_subject=subject,
        )
        return _finish(res)

    platform_keywords = [
        "site down", "service unavailable", "cannot login", "all pages inaccessible",
        "500", "502", "503", "system outage", "platform outage",
    ]
    if any(phrase in combined_lower for phrase in platform_keywords):
        res = DecisionResult(
            safe=True,
            intent="PLATFORM_INCIDENT",
            company=company,
            routing_strategy="BUG_ESCALATION_TEMPLATE",
            retrieval_required=False,
            reason="Platform outage incident detected.",
            normalized_issue=issue,
            normalized_subject=subject,
        )
        return _finish(res)

    THANK_YOU_PHRASES = ["thank you", "thanks a lot", "appreciate it"]
    if len(clean_words) <= 8 and (any(p in combined_lower for p in THANK_YOU_PHRASES) or clean_text in ("thanks", "ty", "thx")):
        res = DecisionResult(
            safe=True,
            intent="THANK_YOU",
            company=company,
            routing_strategy="THANK_YOU_TEMPLATE",
            retrieval_required=False,
            reason="Thank-you message.",
            normalized_issue=issue,
            normalized_subject=subject,
        )
        return _finish(res)

    greeting_words = {"hi", "hello", "hey", "greetings", "good morning", "good evening"}
    if len(clean_words) <= 5 and any(w in greeting_words for w in clean_words):
        res = DecisionResult(
            safe=True,
            intent="GREETING",
            company=company,
            routing_strategy="GREETING_TEMPLATE",
            retrieval_required=False,
            reason="Greeting message.",
            normalized_issue=issue,
            normalized_subject=subject,
        )
        return _finish(res)

    supported_companies = {"hackerrank", "claude", "visa"}
    if company and company.lower() not in supported_companies:
        res = DecisionResult(
            safe=True,
            intent="OUT_OF_SCOPE",
            company=company,
            routing_strategy="OUT_OF_SCOPE_TEMPLATE",
            retrieval_required=False,
            reason=f"Unsupported company: {company}.",
            normalized_issue=issue,
            normalized_subject=subject,
        )
        return _finish(res)

    # -------------------------------------------------------------------------
    # 3. LLM #1 Decision Engine Classification
    # -------------------------------------------------------------------------
    messages = [
        ("system", DECISION_ENGINE_PROMPT),
        ("human", f"Selected Company Hint: {company}\nSubject: {subject}\nIssue: {issue}"),
    ]

    from src.billing.catalog import get_model_price

    price = get_model_price(provider.lower(), model_name.lower())

    try:
        raw_res = llm.generate(messages, response_schema=DecisionResult, stage="decision", provider=provider, model_name=model_name, api_key=api_key)
        dec_obj: DecisionResult = raw_res.response

        # Post-process company detection
        if dec_obj.company_reference_found and dec_obj.company:
            if dec_obj.company.lower() == company.lower():
                dec_obj.routing_status = "COMPANY_VERIFIED"
            else:
                dec_obj.routing_status = "COMPANY_MISMATCH"
        else:
            dec_obj.company_reference_found = False
            dec_obj.routing_status = "NO_COMPANY_REFERENCE"
            dec_obj.company = company

        mapped_strategy = get_strategy_for_intent(dec_obj.intent)
        dec_obj.routing_strategy = mapped_strategy
        dec_obj.retrieval_required = (mapped_strategy == "SUPPORT_RESPONSE")
        dec_obj.normalized_issue = issue
        dec_obj.normalized_subject = subject

        in_tok = raw_res.usage.input_tokens
        out_tok = raw_res.usage.output_tokens
        cost = 0.0
        if price:
            cost = (in_tok / 1_000_000.0) * price.input_per_million + (out_tok / 1_000_000.0) * price.output_per_million

        return _finish(
            dec_obj,
            in_tok=in_tok,
            out_tok=out_tok,
            cost=cost,
            llm_called=True,
        )

    except Exception as e:
        logger.error("Decision Engine LLM failed: %s", e, exc_info=True)
        fallback_res = DecisionResult(
            safe=True,
            intent="PRODUCT_SUPPORT",
            company=company,
            routing_strategy="SUPPORT_RESPONSE",
            retrieval_required=True,
            reason=f"Decision fallback: {e}",
            normalized_issue=issue,
            normalized_subject=subject,
        )
        return _finish(fallback_res)
