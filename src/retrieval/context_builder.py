from __future__ import annotations
from typing import Any

# pyrefly: ignore [missing-import]
from src.config import (MAX_CONTEXT_CHUNKS, MAX_CONTEXT_TOKENS,)


class ContextBuilder:
    """
    Builds the final context passed to the LLM.

    Responsibilities
    ----------------
    - Select the highest-ranked chunks
    - Respect token budget
    - Preserve metadata
    - Produce a clean prompt context
    """

    def build(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:

        context: list[dict[str, Any]] = []

        sources: list[dict[str, str]] = []

        total_tokens = 0

        for result in results:

            metadata = result["metadata"]

            token_estimate = metadata.get(
                "token_estimate",
                0,
            )

            if token_estimate <= 0:
                continue

            if len(context) >= MAX_CONTEXT_CHUNKS:
                break

            if (
                total_tokens + token_estimate
                > MAX_CONTEXT_TOKENS
            ):
                break

            context.append(
                {
                    "company": metadata["company"],
                    "product_area": metadata["product_area"],
                    "title": metadata["title"],
                    "section": metadata["section"],
                    "url": metadata["url"],
                    "text": result["text"],
                }
            )

            sources.append(
                {
                    "company": metadata["company"],
                    "title": metadata["title"],
                    "url": metadata["url"],
                }
            )

            total_tokens += token_estimate

        return {
            "context": context,
            "sources": sources,
            "num_chunks": len(context),
            "token_estimate": total_tokens,
        }
