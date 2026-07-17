from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseReranker(ABC):
    """
    Abstract base class for all rerankers.

    Every reranker receives the user query together
    with the retrieved candidate chunks and returns
    a reranked list ordered by relevance.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Rerank retrieval results.

        Parameters
        ----------
        query
            User search query.

        results
            Candidate retrieval results.

        top_k
            Number of chunks to return.

        Returns
        -------
        list[dict[str, Any]]
            Reranked retrieval results.
        """
        raise NotImplementedError