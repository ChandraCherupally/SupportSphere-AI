import logging
import time
from typing import Any

logger = logging.getLogger("SupportSphere")


def log_ticket_summary(res: Any) -> None:
    """Log a clean, structured ticket execution summary block."""
    ticket_id = getattr(res, "ticket_id", "N/A") or "N/A"
    if ticket_id == "N/A" and hasattr(res, "get"):
        ticket_id = res.get("ticket_id", "N/A")

    start_time = getattr(res, "start_time", None)
    if start_time is None and hasattr(res, "get"):
        start_time = res.get("start_time")
    if not start_time:
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")

    def get_attr(obj: Any, key: str, default: Any = "N/A") -> Any:
        val = None
        if hasattr(obj, key):
            val = getattr(obj, key)
        elif hasattr(obj, "get"):
            val = obj.get(key)
        return val if val is not None else default

    subject = get_attr(res, "subject")
    issue = get_attr(res, "issue")
    selected_company = get_attr(res, "selected_company")
    detected_company = get_attr(res, "detected_company", None)
    verified_company = get_attr(res, "verified_company")
    company_reference_found = get_attr(res, "company_reference_found", False)
    reference_confidence = get_attr(res, "reference_confidence", 1.0)
    routing_status = get_attr(res, "routing_status", "NO_COMPANY_REFERENCE")
    retrieval_required = get_attr(res, "retrieval_required", True)
    routing_reason = get_attr(res, "routing_reason")
    search_mode = get_attr(res, "search_mode")
    reranker = get_attr(res, "reranker")

    # Read execution_trace list
    raw_trace = get_attr(res, "execution_trace", []) or []
    if isinstance(raw_trace, list):
        obs: dict = {r["stage"]: r for r in raw_trace if isinstance(r, dict) and "stage" in r}
    elif isinstance(raw_trace, dict):
        obs = raw_trace
    else:
        obs = {}

    # DECISION ENGINE
    de = obs.get("decision_engine", {})
    de_mode = de.get("execution_mode", "LLM")
    de_provider = de.get("provider", get_attr(res, "decision_provider", "google"))
    de_model = de.get("model", get_attr(res, "decision_model", "gemini-2.5-flash-lite"))
    de_in_tok = de.get("input_tokens", 0)
    de_out_tok = de.get("output_tokens", 0)
    de_latency = de.get("latency_ms", de.get("latency", 0.0))
    de_cost = de.get("cost", 0.0)

    # RETRIEVAL
    ret = obs.get("retrieval", {})
    ret_latency = ret.get("latency_ms", ret.get("latency", 0.0))
    ret_candidates = ret.get("candidates", 0)
    ret_cost = ret.get("cost", 0.0)

    # RETRIEVAL VALIDATION
    rv = obs.get("retrieval_validation", {})
    rv_decision = rv.get("decision", "ALLOW")
    rv_latency = rv.get("latency_ms", rv.get("latency", 0.0))

    # GENERATION
    gen = obs.get("generation", {})
    gen_provider = gen.get("provider", "N/A")
    gen_model = gen.get("model", "N/A")
    gen_in_tok = gen.get("input_tokens", 0)
    gen_out_tok = gen.get("output_tokens", 0)
    gen_latency = gen.get("latency_ms", gen.get("latency", 0.0))
    gen_cost = gen.get("cost", 0.0)

    # RESPONSE VALIDATION
    rv_out = obs.get("response_validation", {})
    rv_out_decision = rv_out.get("decision", "PASS")
    rv_out_latency = rv_out.get("latency_ms", rv_out.get("latency", 0.0))

    # SUMMARY
    sum_req_type = "retrieval_based" if retrieval_required else "direct"
    sum_status = "resolved" if not get_attr(res, "blocked", False) else "blocked"
    if get_attr(res, "regenerated", False):
        sum_status = "regenerated"

    total_tokens = de_in_tok + de_out_tok + gen_in_tok + gen_out_tok
    total_cost = de_cost + ret_cost + gen_cost
    total_latency = de_latency + ret_latency + rv_latency + gen_latency + rv_out_latency
    warnings = get_attr(res, "warnings", [])
    warnings_str = ", ".join(warnings) if warnings else "None"

    summary_lines = [
        "================================================================================",
        "SupportSphere AI | Ticket ID: {} | Execution Started: {}".format(ticket_id, start_time),
        "================================================================================",
        "Ticket: {}".format(subject),
        "Subject: {}".format(subject),
        "Issue: {}".format(str(issue).replace("\n", " ")),
        "Selected Company: {}".format(selected_company),
        "Decision Model: {}".format(de_model),
        "Generation Model: {}".format(gen_model),
        "Search Mode: {}".format(search_mode),
        "Reranker: {}".format(reranker),
        "------------------------------------------------------------",
        "1. DECISION ENGINE (LLM #1)",
        "Execution Mode: {}".format(de_mode),
        "Provider: {} | Model: {}".format(de_provider, de_model),
        "Intent: {}".format(get_attr(res, "intent", "PRODUCT_SUPPORT")),
        "Routing Strategy: {}".format(get_attr(res, "routing_strategy", "SUPPORT_RESPONSE")),
        "Selected Company: {}".format(selected_company),
        "Company Reference Found: {}".format(company_reference_found),
        "Reference Confidence: {:.2f}".format(reference_confidence),
        "Routing Status: {}".format(routing_status),
        "Detected Company: {}".format(detected_company or "None"),
        "Verified Company: {}".format(verified_company),
        "Retrieval Required: {}".format(retrieval_required),
        "Routing Reason: {}".format(routing_reason),
        "Tokens (In/Out): {} / {}".format(de_in_tok, de_out_tok),
        "Cost: ${:.6f} | Latency: {:.1f}ms".format(de_cost, de_latency),
        "------------------------------------------------------------",
        "2. RETRIEVAL",
        "Collection: {}".format(verified_company),
        "Search Mode: {}".format(search_mode),
        "Candidates Retrieved: {}".format(ret_candidates),
        "Cost: ${:.6f} | Latency: {:.1f}ms".format(ret_cost, ret_latency),
        "------------------------------------------------------------",
        "3. RETRIEVAL VALIDATION (Rules-Only)",
        "Decision: {}".format(rv_decision),
        "Cost: $0.000000 | Latency: {:.1f}ms".format(rv_latency),
        "------------------------------------------------------------",
        "4. GENERATION ENGINE (LLM #2)",
        "Provider: {} | Model: {}".format(gen_provider, gen_model),
        "Tokens (In/Out): {} / {}".format(gen_in_tok, gen_out_tok),
        "Cost: ${:.6f} | Latency: {:.1f}ms".format(gen_cost, gen_latency),
        "------------------------------------------------------------",
        "5. RESPONSE VALIDATION (Rules-Only)",
        "Decision: {}".format(rv_out_decision),
        "Cost: $0.000000 | Latency: {:.1f}ms".format(rv_out_latency),
        "------------------------------------------------------------",
        "SUMMARY",
        "Request Type: {}".format(sum_req_type),
        "Status: {}".format(sum_status),
        "Total Tokens: {}".format(total_tokens),
        "Total Cost: ${:.6f}".format(total_cost),
        "Total Latency: {:.1f}ms".format(total_latency),
        "Warnings: {}".format(warnings_str),
        "================================================================================",
    ]
    for line in summary_lines:
        logger.info(line)
