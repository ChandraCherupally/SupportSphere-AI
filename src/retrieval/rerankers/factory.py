from __future__ import annotations

import logging

from src.retrieval.rerankers.base import BaseReranker
from src.retrieval.rerankers.noop_reranker import NoOpReranker
from src.retrieval.rerankers.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.rerankers.llm_reranker import LLMReranker
from src.retrieval.rerankers.flashrank_reranker import FlashRankReranker

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
        reranker = (reranker.strip().lower())
        logger.info("Creating reranker: %s",reranker,)

        #
        # Already created?
        #
        if reranker in cls._instances:
            logger.debug("Using cached reranker: %s",reranker,)
            return cls._instances[reranker]

        logger.info("Creating reranker: %s",reranker,)

        try:
            if reranker == "cross_encoder":
                logger.info("Creating reranker: %s",reranker,)
                instance = CrossEncoderReranker()
            elif reranker == "llm":
                logger.info("Creating reranker: %s",reranker,)
                instance = LLMReranker()
            elif reranker == "flashrank":
                logger.info("Creating reranker: %s",reranker,)
                instance = FlashRankReranker()
            else:
                logger.info("Creating reranker: %s",reranker,)
                instance =  NoOpReranker()
            
            #
            # Cache instance.
            #
            cls._instances[reranker] = instance
            return instance

        except Exception as exc:
            logger.exception("Failed to create reranker: %s", exc)
            if "none" not in cls._instances:
                cls._instances["none"] = NoOpReranker()
            return cls._instances["none"]

