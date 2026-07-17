from __future__ import annotations

import logging
from typing import Any

from google import genai
from google.genai import types


import src.config as config
from src.retrieval.rerankers.base import BaseReranker

logger = logging.getLogger(__name__)

from src.retrieval.rerankers.schemas import (
    RerankerResponse,
)


class LLMReranker(BaseReranker):
    """
    Gemini Flash based reranker.

    Uses Gemini to reorder candidate
    retrieval results according to
    semantic relevance.
    """

    def __init__(self):
        self.client = genai.Client(api_key=config.GOOGLE_API_KEY)
        self.model_name = (config.LLM_RERANKER_MODEL)

    def rerank(self, query: str, results: list[dict[str, Any]], top_k: int,) -> list[dict[str, Any]]:
        """
        Rerank retrieved chunks using Gemini.
        """

        logger.info("Reranker selected: Gemini Flash")

        if not results:
            logger.warning("No retrieval results available.")
            return []

        try:
            prompt = self._build_prompt(query=query,results=results,)

            logger.info("Sending %d chunks to Gemini for reranking.",len(results),)

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=RerankerResponse,
                ),
            )

            ranked_ids = response.parsed.ranking
            logger.info("Gemini returned %d ranked chunks.", len(ranked_ids),)

            #
            # Lookup table.
            #
            lookup = {result["id"]: result for result in results}
            reranked_results = []

            #
            # Preserve Gemini ranking.
            #
            for chunk_id in ranked_ids:
                if chunk_id in lookup:
                    reranked_results.append(lookup.pop(chunk_id))

            #
            # Append remaining chunks
            # in original RRF order.
            #
            reranked_results.extend(lookup.values())
            logger.info("Gemini reranking completed.")
            return reranked_results[:top_k]

        except Exception as exc:
            logger.exception("Gemini reranking failed: %s",exc,)
            logger.warning("Returning original RRF ranking.")
            return results[:top_k]


    @staticmethod
    def _build_prompt(query: str,results: list[dict[str, Any]],) -> str:
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

