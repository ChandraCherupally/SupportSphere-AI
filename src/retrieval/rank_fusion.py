from __future__ import annotations

from typing import Any


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF).

    Combines multiple ranked retrieval lists into a
    single ranking using reciprocal rank fusion.

    RRF is rank-based rather than score-based, making it
    robust across heterogeneous retrieval systems such as:

    - BM25
    - Dense Vector Search
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        *,
        bm25: list[dict[str, Any]],
        vector: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Merge BM25 and Vector Search rankings.

        Parameters
        ----------
        bm25
            Ranked BM25 results.

        vector
            Ranked semantic search results.

        Returns
        -------
        list
            RRF-ranked retrieval results.
        """

        fused: dict[str, dict[str, Any]] = {}

        retrieval_lists = (
            ("bm25", bm25),
            ("vector", vector),
        )

        for source_name, ranked_results in retrieval_lists:

            for rank, result in enumerate(ranked_results, start=1):

                chunk_id = result["id"]

                if chunk_id not in fused:

                    fused[chunk_id] = result.copy()

                    fused[chunk_id]["rrf_score"] = 0.0

                    fused[chunk_id]["retrieval_sources"] = []

                fused[chunk_id]["rrf_score"] += (
                    1.0 / (self.k + rank)
                )

                if source_name not in fused[chunk_id]["retrieval_sources"]:

                    fused[chunk_id]["retrieval_sources"].append(
                        source_name
                    )

        return sorted(
            fused.values(),
            key=lambda item: (
                item["rrf_score"],
                item.get("vector_score", 0.0),
                item.get("bm25_score", 0.0),
            ),
            reverse=True,
        )