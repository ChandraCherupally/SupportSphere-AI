from __future__ import annotations

from typing import Any

from src.config import (
    BM25_TOP_K,
    VECTOR_TOP_K,
    FINAL_TOP_K,
)
from src.retrieval.bm25_index import BM25Index
from src.retrieval.vector_search import VectorSearch
from src.retrieval.rank_fusion import ReciprocalRankFusion


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

    def __init__(self):

        self.bm25 = BM25Index()
        self.bm25.load()

        self.vector = VectorSearch()

        self.rrf = ReciprocalRankFusion()

    def search(
        self,
        query: str,
        company: str | None = None,
        filters: dict[str, Any] | None = None,
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

        bm25_results = self.bm25.search(
            query=query,
            top_k=BM25_TOP_K,
            company=company,
        )

        # -------------------------------------------------------
        # Vector Search
        #
        # Company metadata filtering.
        # Better precision.
        # -------------------------------------------------------

        vector_results = self.vector.search(
            query=query,
            top_k=VECTOR_TOP_K,
            filters=vector_filters,
        )

        fused = self.rrf.fuse(
            bm25=bm25_results,
            vector=vector_results,
        )

        return fused[:FINAL_TOP_K]

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