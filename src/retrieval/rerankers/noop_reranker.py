from __future__ import annotations

import logging
from typing import Any
from src.retrieval.rerankers.base import BaseReranker
logger = logging.getLogger(__name__)



class NoOpReranker(BaseReranker):
    """
    No-operation reranker.

    This reranker performs no additional ranking after
    Reciprocal Rank Fusion (RRF). It simply returns the
    highest-ranked results produced by the retrieval
    pipeline.

    This is the default reranker used when reranking is
    disabled.
    """

    def rerank(self, query: str, results: list[dict[str, Any]], top_k: int,) -> list[dict[str, Any]]:
        """
        Return the original retrieval results.

        Parameters
        ----------
        query
            User search query.
            (Unused for this implementation.)

        results
            Results returned by Reciprocal Rank Fusion.

        top_k
            Number of chunks to return.

        Returns
        -------
        list[dict[str, Any]]
            Top-K retrieval results.
        """

        logger.info("Reranker selected: None")

        try:
            if not results:
                logger.warning("No retrieval results available for reranking.")
                return []

            reranked_results = results[:top_k]
            logger.info("Returning %d chunks without reranking.", len(reranked_results))
            return reranked_results

        except Exception as exc:
            logger.exception("Unexpected error in NoOpReranker: %s",exc)
            #
            # Fail safely.
            #
            return results[:top_k] if results else []