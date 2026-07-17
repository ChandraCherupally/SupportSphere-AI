from __future__ import annotations
import logging
from typing import Any
import src.config as config
from src.retrieval.rerankers.base import BaseReranker
logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Local Cross-Encoder reranker.

    Uses a sentence-transformers CrossEncoder to
    rescore retrieved chunks.

    The model is loaded lazily to avoid importing
    Torch unless this reranker is explicitly used.
    """

    def __init__(self, model_name: str | None = None,):
        self.model_name = (model_name or config.CROSS_ENCODER_MODEL)
        self.model: Any | None = None

    def _load_model(self) -> bool:
        """
        Lazily load the CrossEncoder model.

        Returns
        -------
        bool
            True if model loaded successfully,
            otherwise False.
        """

        if self.model is not None:
            return True

        try:

            logger.info("Loading CrossEncoder model: %s", self.model_name,)

            #
            # Lazy import.
            #
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info("CrossEncoder loaded successfully.")
            return True

        except Exception as exc:
            logger.exception("Failed to load CrossEncoder model: %s", exc,)
            return False

    def rerank(self, query: str, results: list[dict[str, Any]],top_k: int,) -> list[dict[str, Any]]:
        """
        Rerank retrieval results using
        CrossEncoder relevance scores.
        """

        logger.info("Reranker selected: Cross Encoder")

        if not results:
            logger.warning("No retrieval results available.")
            return []

        #
        # Load model.
        #
        if not self._load_model():
            logger.warning("Falling back to RRF results.")
            return results[:top_k]

        try:
            #
            # Build query-document pairs.
            #
            pairs = [(query, result["text"]) for result in results]
            logger.info("Scoring %d retrieved chunks.", len(pairs),)
            scores = self.model.predict(pairs, show_progress_bar=False)

            #
            # Attach scores.
            #
            for result, score in zip(results,scores,):
                result["rerank_score"] = float(score)
            #
            # Sort by score.
            #
            reranked_results = sorted(results, key=lambda item: item["rerank_score"],reverse=True,)
            logger.info("CrossEncoder reranking completed.")

            return reranked_results[:top_k]

        except Exception as exc:
            logger.exception("CrossEncoder reranking failed: %s",exc,)
            logger.warning("Returning original RRF ranking.")
            return results[:top_k]
