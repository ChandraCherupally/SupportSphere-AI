from __future__ import annotations

import logging
from typing import Any
from flashrank import Ranker, RerankRequest
import src.config as config
from src.retrieval.rerankers.base import BaseReranker

logger = logging.getLogger(__name__)

class FlashRankReranker(BaseReranker):
    """
    FlashRank reranker.

    Uses FlashRank's ONNX cross-encoder
    for fast local reranking.
    """

    def __init__(self):
        self.model = None

    def _load_model(self) -> bool:
        
        if self.model is not None:
            return True

        try:

            logger.info("Loading FlashRank model: %s",config.FLASHRANK_MODEL,)
            self.model = Ranker(model_name=config.FLASHRANK_MODEL, 
                                max_length=config.FLASHRANK_MAX_LENGTH,)
            logger.info("FlashRank loaded successfully.")

            return True

        except Exception as exc:
            logger.exception("Unable to load FlashRank: %s",exc,)
            return False

    def rerank(self, query: str, results: list[dict[str, Any]], top_k: int,) -> list[dict[str, Any]]:

        logger.info("Reranker selected: FlashRank")

        if not results:
            return []

        if not self._load_model():
            return results[:top_k]

        try:
            passages = []
            for result in results:

                passages.append(
                    {"id": result["id"],
                    "text": result["text"],
                    "meta": result.get("metadata",{},),}
                )

            request = RerankRequest(query=query,passages=passages,)
            ranked = self.model.rerank(request)
            lookup = {r["id"]: r for r in results}
            reranked = []

            for item in ranked:
                chunk_id = item["id"]

                if chunk_id in lookup:
                    lookup[chunk_id]["rerank_score"] = item.get("score", None,)
                    reranked.append(lookup.pop(chunk_id))

            reranked.extend(lookup.values())
            logger.info("FlashRank reranking completed.")
            return reranked[:top_k]

        except Exception as exc:
            logger.exception("FlashRank failed: %s", exc,)
            return results[:top_k]

