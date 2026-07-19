from __future__ import annotations

import logging
from typing import Any

import src.config as config
from src.retrieval.rerankers.base import BaseReranker
from src.ai.client import LLMClient
from src.retrieval.rerankers.schemas import RerankerResponse

logger = logging.getLogger(__name__)


class LLMReranker(BaseReranker):
    """
    LLM-based reranker.

    Reuses the selected Decision Model to reorder candidate
    retrieval results according to semantic relevance.
    """

    def __init__(self):
        self.client = LLMClient()
        self.input_tokens = 0
        self.output_tokens = 0

    def rerank(self, query: str, results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """
        Rerank retrieved chunks using the Decision Model.
        """
        logger.info("Reranker selected: LLM Reranker")

        if not results:
            logger.warning("No retrieval results available.")
            return []

        self.input_tokens = 0
        self.output_tokens = 0

        try:
            prompt = self._build_prompt(query=query, results=results)
            messages = [
                ("system", "You are a helpful assistant that ranks support documentation chunks."),
                ("human", prompt)
            ]

            logger.info("Sending %d chunks to LLM for reranking.", len(results))

            result = self.client.generate(
                messages=messages,
                response_schema=RerankerResponse,
                stage="decision"
            )

            # Record token usage
            self.input_tokens = result.usage.input_tokens
            self.output_tokens = result.usage.output_tokens

            ranked_ids = result.response.ranking
            logger.info("LLM reranker returned %d ranked chunks.", len(ranked_ids))

            # Lookup table
            lookup = {result["id"]: result for result in results}
            reranked_results = []

            # Preserve LLM ranking
            for chunk_id in ranked_ids:
                if chunk_id in lookup:
                    reranked_results.append(lookup.pop(chunk_id))

            # Append remaining chunks in original RRF order
            reranked_results.extend(lookup.values())
            logger.info("LLM reranking completed.")
            return reranked_results[:top_k]

        except Exception as exc:
            logger.exception("LLM reranking failed: %s", exc)
            logger.warning("Returning original RRF ranking.")
            return results[:top_k]

    @staticmethod
    def _build_prompt(query: str, results: list[dict[str, Any]]) -> str:
        """
        Build Gemini reranking prompt.
        """
        chunks = []
        for result in results:
            metadata = result["metadata"]
            chunks.append(
                f"""
                Chunk ID: {result["id"]}

                Company:
                {metadata.get("company", "")}

                Product Area:
                {metadata.get("product_area", "")}

                Title:
                {metadata.get("title", "")}

                Section:
                {metadata.get("section", "")}

                Content:
                {result["text"]}

                ------------------------------------------------------------
                """
            )

        return f"""
                You are an expert retrieval reranker for a customer support AI assistant.

                Your task is to rank the candidate support documentation
                from MOST relevant to LEAST relevant.

                Ranking Rules

                1. Directly answers the user's question.
                2. Matches the correct company.
                3. Matches the correct product area.
                4. Contains actionable support instructions.
                5. Prefer complete answers over partial answers.
                6. Prefer official documentation over indirect references.

                Important Rules

                • Do NOT generate new Chunk IDs.
                • Do NOT remove any Chunk IDs.
                • Return every Chunk ID exactly once.

                User Query

                {query}

                Candidate Chunks

                {"".join(chunks)}
                """
