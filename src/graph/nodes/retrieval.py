"""
Retrieval Node.

Executes sparse BM25 + dense Vector hybrid search to fetch relevant context chunks.
Appends a StageRecord for "retrieval".
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from src.graph.state import SupportState
from src.graph.trace import ExecutionTrace, StageRecord
from src.retrieval.retriever import Retriever

logger = logging.getLogger("SupportSphere")

_retriever_cache = {}


def get_retriever(reranker: str, search_mode: str) -> Retriever:
    """Return a cached Retriever singleton."""
    import src.config as src_config

    api_key = src_config.GOOGLE_API_KEY or ""
    key = (reranker, search_mode, api_key)
    if key not in _retriever_cache:
        _retriever_cache[key] = Retriever(reranker=reranker, search_mode=search_mode)
    return _retriever_cache[key]


def retrieve_node(state: SupportState) -> Dict[str, Any]:
    """Retrieve top-k relevant knowledge base documentation chunks."""
    trace: ExecutionTrace = state.get("execution_trace")
    if trace is None:
        trace = ExecutionTrace()
        state["execution_trace"] = trace

    if not state.get("retrieval_required", True) or state.get("blocked", False):
        trace.append(
            StageRecord(
                stage="retrieval",
                execution_mode="Skipped",
                provider="N/A",
                model="N/A",
                llm_called=False,
                decision="PASS",
            )
        )
        return {
            "context": [],
            "retrieved_context": [],
            "retrieved_chunks": [],
            "num_chunks": 0,
            "token_estimate": 0,
            "retrieval_trace": {},
            "retrieval_latency_ms": 0.0,
            "execution_trace": trace,
        }

    t_start = time.perf_counter()
    logger.debug("Executing Retrieval Node.")

    reranker_name = state.get("reranker", "none")
    search_mode = state.get("search_mode", "hybrid")
    retriever = get_retriever(reranker=reranker_name, search_mode=search_mode)

    verified_co = state.get("verified_company", state.get("company", ""))
    retrieval = retriever.retrieve(
        issue=state["issue"],
        subject=state.get("subject", ""),
        company=verified_co,
    )
    retrieval["retrieved_context"] = [chunk["text"] for chunk in retrieval["context"]]

    import src.config as src_config

    r_trace = retrieval.get("retrieval_trace", {})
    r_trace["selected_company"] = state.get("selected_company", state.get("company", ""))
    r_trace["verified_company"] = verified_co
    r_trace["collection_used"] = "support-sphere-ai"
    r_trace["top_k"] = getattr(src_config, "FINAL_TOP_K", 5)

    from src.billing.catalog import PINECONE_QUERY_COST, get_embedding_price

    emb_price = get_embedding_price("google", "gemini-embedding-001")
    estimated_query_tokens = len(state.get("issue", "")) // 4
    emb_cost = 0.0
    if emb_price:
        emb_cost = (estimated_query_tokens / 1_000_000.0) * emb_price.input_per_million
    retrieval_cost = emb_cost + PINECONE_QUERY_COST

    node_latency = (time.perf_counter() - t_start) * 1000.0
    duplicates_removed = max(0, r_trace.get("merged_count", 0) - r_trace.get("unique_count", 0))

    trace.append(
        StageRecord(
            stage="retrieval",
            execution_mode=f"{search_mode.title()} Search",
            provider="Google/Pinecone",
            model="gemini-embedding-001",
            llm_called=False,
            latency_ms=node_latency,
            cost=retrieval_cost,
            decision="PASS",
            extra={
                "collection": verified_co,
                "candidates": r_trace.get("merged_count", 0),
                "duplicates_removed": duplicates_removed,
                "final_context_chunks": len(retrieval["context"]),
                "estimated_tokens": retrieval.get("token_estimate", 0),
            },
        )
    )

    retrieval["retrieval_latency_ms"] = node_latency
    retrieval["execution_trace"] = trace
    return retrieval
