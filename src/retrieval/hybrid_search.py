"""
Hybrid Retrieval Pipeline combining BM25 and Vector Search.
Supports search mode configuration, deduplication (including URL-cap), and loop fallback to guarantee Top-K.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import src.config as src_config
from src.retrieval.bm25_index import BM25Index
from src.retrieval.rank_fusion import ReciprocalRankFusion
from src.retrieval.rerankers.factory import RerankerFactory
from src.retrieval.vector_search import VectorSearch

logger = logging.getLogger(__name__)


class HybridSearch:
    """
    Hybrid Retrieval Pipeline.
    """

    def __init__(self, reranker: str = "none", search_mode: str = "hybrid") -> None:
        self.bm25 = BM25Index()
        self.bm25.load()

        self.vector = VectorSearch()

        self.rrf = ReciprocalRankFusion()
        self.reranker = RerankerFactory.create(reranker)
        self.search_mode = search_mode.lower()

    def search(
        self,
        query: str,
        company: str | None = None,
        filters: dict[str, Any] | None = None,
        clean_query: str | None = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute retrieval matching the configured strategy.

        Returns:
            Tuple of (Ranked results, Trace dictionary of step stats).
        """
        vector_filters = self._build_filters(
            company=company,
            filters=filters,
        )

        final_top_k = getattr(src_config, "FINAL_TOP_K", 10)
        max_chunks_per_url = getattr(src_config, "MAX_CHUNKS_PER_URL", 2)

        # Resolve the active search mode (priority: parameter -> config -> default)
        active_mode = self.search_mode
        if not active_mode:
            active_mode = getattr(src_config, "SEARCH_MODE", "hybrid").lower()
        active_mode = active_mode.lower()

        if active_mode not in {"bm25", "vector", "hybrid"}:
            active_mode = "hybrid"

        logger.info(
            "Executing retrieval for query: '%s' | Company: '%s' | Configured Top-K: %d | Search Mode: %s",
            query, company, final_top_k, active_mode
        )

        # Helper to query candidates from search backends
        def get_candidates(limit: int) -> Tuple[List[Dict[str, Any]], int, int, str]:
            bm25_res = []
            vector_res = []

            if active_mode in {"bm25", "hybrid"}:
                bm25_res = self.bm25.search(
                    query=query,
                    top_k=limit,
                    company=company,
                )

            if active_mode in {"vector", "hybrid"}:
                vector_res = self.vector.search(
                    query=clean_query or query,
                    top_k=limit,
                    filters=vector_filters,
                )

            if active_mode == "hybrid":
                fused_res = self.rrf.fuse(
                    bm25=bm25_res,
                    vector=vector_res,
                )
                strategy = "Hybrid (BM25 + Dense Vector Search via RRF)"
            elif active_mode == "bm25":
                fused_res = bm25_res
                strategy = "Keyword Search (BM25 Only)"
            else:
                fused_res = vector_res
                strategy = "Semantic Search (Dense Vector Only)"

            return fused_res, len(bm25_res), len(vector_res), strategy

        # Helper to perform deduplication & URL-cap filtering
        def deduplicate(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            seen_ids = set()
            seen_texts = set()
            url_counts: Dict[str, int] = {}
            unique = []

            for r in results:
                chunk_id = r.get("id") or r.get("chunk_id")
                text = (r.get("text") or "").strip()
                url = (r.get("metadata", {}).get("url") or "").lower().split("?")[0]

                if chunk_id and chunk_id in seen_ids:
                    continue
                if text in seen_texts:
                    continue
                if url and url_counts.get(url, 0) >= max_chunks_per_url:
                    continue

                if chunk_id:
                    seen_ids.add(chunk_id)
                if text:
                    seen_texts.add(text)
                if url:
                    url_counts[url] = url_counts.get(url, 0) + 1

                unique.append(r)

            return unique

        # Run retrieval loop to guarantee Top-K unique chunks
        current_limit = max(20, final_top_k * 2)
        max_attempts = 3

        candidates: List[Dict[str, Any]] = []
        unique_candidates: List[Dict[str, Any]] = []
        bm25_count = 0
        vector_count = 0
        strategy_executed = ""

        for attempt in range(max_attempts):
            candidates, bm25_count, vector_count, strategy_executed = get_candidates(current_limit)
            unique_candidates = deduplicate(candidates)

            # Break if we have enough unique candidates to satisfy Top-K
            if len(unique_candidates) >= final_top_k:
                logger.debug(
                    "Attempt %d succeeded. Found %d unique candidates (Limit: %d).",
                    attempt + 1, len(unique_candidates), current_limit
                )
                break

            # Break if backend returned fewer candidates than limit (exhausted corpus)
            if len(candidates) < current_limit:
                logger.debug(
                    "Attempt %d: corpus exhausted. Only %d raw candidates returned (Limit: %d).",
                    attempt + 1, len(candidates), current_limit
                )
                break

            logger.debug(
                "Attempt %d: only %d unique candidates found. Scaling search candidate limit.",
                attempt + 1, len(unique_candidates)
            )
            current_limit *= 2

        # Rerank on unique candidates
        reranked = self.reranker.rerank(
            query=clean_query or query,
            results=unique_candidates,
            top_k=final_top_k,
        )

        # Telemetry logs using logger.debug
        logger.debug("--- Retrieval Step Performance ---")
        logger.debug("Selected Search Mode: %s", active_mode)
        logger.debug("Search Backend Executed: %s", strategy_executed)
        logger.debug("Number of BM25 Results: %d", bm25_count)
        logger.debug("Number of Vector Results: %d", vector_count)
        logger.debug("Number after Merge: %d", len(candidates))
        logger.debug("Number after Deduplication: %d", len(unique_candidates))
        logger.debug("Number after Reranking: %d", len(reranked))
        logger.debug("Final Returned Chunks: %d", len(reranked))

        trace = {
            "search_mode": active_mode,
            "strategy_executed": strategy_executed,
            "bm25_initial_count": bm25_count,
            "vector_initial_count": vector_count,
            "merged_count": len(candidates),
            "unique_count": len(unique_candidates),
            "reranked_count": len(reranked),
            "final_returned_count": len(reranked),
        }

        return reranked, trace

    @staticmethod
    def _build_filters(
        company: str | None,
        filters: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """
        Build metadata filters.
        """
        metadata_filters = dict(filters or {})
        company = (company or "").strip().lower()

        if company and company != "none":
            metadata_filters["company"] = company

        return metadata_filters or None