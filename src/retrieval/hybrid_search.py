from __future__ import annotations

import logging
from typing import Any
import src.config as src_config

logger = logging.getLogger(__name__)
from src.retrieval.bm25_index import BM25Index
from src.retrieval.vector_search import VectorSearch
from src.retrieval.rank_fusion import ReciprocalRankFusion
from src.retrieval.rerankers.factory import (
    RerankerFactory,
)


class HybridSearch:
    """
    Hybrid Retrieval Pipeline.

    Retrieval Flow

        BM25
           \
            \
             RRF ----> Final Ranking
            /
           /
    Vector Search

    Design Goals
    ------------

    • High recall
    • Low latency
    • Production simplicity
    """

    def __init__(self,reranker: str = "none",):

        self.bm25 = BM25Index()
        self.bm25.load()

        self.vector = VectorSearch()

        self.rrf = ReciprocalRankFusion()
        self.reranker = RerankerFactory.create(reranker)

    def search(
        self,
        query: str,
        company: str | None = None,
        filters: dict[str, Any] | None = None,
        clean_query: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute hybrid retrieval.

        Parameters
        ----------
        query
            Search query.

        company
            Ticket company.

            HackerRank
            Claude
            Visa
            None

        filters
            Additional metadata filters.

        clean_query
            A clean natural-language query (optional) without keyword stuffing.
            Highly recommended for vector embeddings and reranking.

        Returns
        -------
        Ranked retrieval results.
        """

        vector_filters = self._build_filters(
            company=company,
            filters=filters,
        )

        # -------------------------------------------------------
        # BM25
        #
        # No metadata filtering.
        # Better recall.
        # -------------------------------------------------------

        # Get dynamic configuration variables
        final_top_k = getattr(src_config, "FINAL_TOP_K", 10)
        bm25_top_k = getattr(src_config, "BM25_TOP_K", 20)
        vector_top_k = getattr(src_config, "VECTOR_TOP_K", 20)

        # Scale intermediate top_k values to prevent bottlenecks in rerankers
        bm25_top_k = max(bm25_top_k, final_top_k * 2)
        vector_top_k = max(vector_top_k, final_top_k * 2)

        logger.info(f"Executing retrieval for query: '{query}' | Company: '{company}' | Configured Top-K: {final_top_k}")

        bm25_results = self.bm25.search(
            query=query,
            top_k=bm25_top_k,
            company=company,
        )

        # -------------------------------------------------------
        # Vector Search
        #
        # Company metadata filtering.
        # Better precision.
        # -------------------------------------------------------

        vector_results = self.vector.search(
            query=clean_query or query,
            top_k=vector_top_k,
            filters=vector_filters,
        )

        fused = self.rrf.fuse(
            bm25=bm25_results,
            vector=vector_results,
        )

        reranked = self.reranker.rerank(
            query=clean_query or query,
            results=fused,
            top_k=final_top_k,
        )

        logger.info(f"Retrieval complete. Fused and reranked to {len(reranked)} chunks.")
        return reranked



    @staticmethod
    def _build_filters(
        company: str | None,
        filters: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """
        Build metadata filters.

        Company filtering is only applied
        when company is explicitly known.

        company == "None"

        →

        No filtering.
        """

        metadata_filters = dict(filters or {})

        company = (company or "").strip().lower()

        if company and company != "none":

            metadata_filters["company"] = company

        return metadata_filters or None