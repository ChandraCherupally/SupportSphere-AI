from __future__ import annotations

import logging

from src.retrieval.rerankers.base import BaseReranker
from src.retrieval.rerankers.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.rerankers.flashrank_reranker import FlashRankReranker
from src.retrieval.rerankers.llm_reranker import LLMReranker
from src.retrieval.rerankers.noop_reranker import NoOpReranker

logger = logging.getLogger(__name__)


class RerankerFactory:
    """
    Factory for creating reranker implementations.

    Rerankers are cached after their first creation
    so models are loaded only once.
    """

    _instances: dict[str, BaseReranker] = {}

    @classmethod
    def create(cls, reranker: str) -> BaseReranker:
        reranker_name = reranker.strip().lower()

        # Already cached?
        if reranker_name in cls._instances:
            logger.debug("Using cached reranker: %s", reranker_name)
            return cls._instances[reranker_name]

        logger.info("Creating reranker: %s", reranker_name)

        try:
            if reranker_name == "cross_encoder":
                instance = CrossEncoderReranker()
            elif reranker_name == "llm":
                instance = LLMReranker()
            elif reranker_name == "flashrank":
                instance = FlashRankReranker()
            else:
                instance = NoOpReranker()

            cls._instances[reranker_name] = instance
            return instance

        except Exception as exc:
            logger.exception("Failed to create reranker: %s", exc)
            return NoOpReranker()
