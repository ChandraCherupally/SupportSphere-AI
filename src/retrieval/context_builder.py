from __future__ import annotations
from typing import Any

# pyrefly: ignore [missing-import]
import src.config as src_config
from src.ai.models import RetrievedChunk

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

        retrieved_chunks: list[RetrievedChunk] = []

        total_tokens = 0

        # URL-level deduplication: keep at most 2 chunks per source URL.
        # This directly improves RAGAS Context Precision by avoiding
        # retrieval bloat where many low-relevance sections from the
        # same page crowd out truly relevant chunks from other URLs.
        max_chunks_per_url = getattr(src_config, "MAX_CHUNKS_PER_URL", 2)
        url_counts: dict[str, int] = {}

        for result in results:

            metadata = result["metadata"]

            token_estimate = metadata.get(
                "token_estimate",
                0,
            )

            if token_estimate <= 0:
                continue

            max_chunks = getattr(src_config, "FINAL_TOP_K", getattr(src_config, "MAX_CONTEXT_CHUNKS", 10))
            max_tokens = getattr(src_config, "MAX_CONTEXT_TOKENS", 6000)
            if len(context) >= max_chunks:
                break

            if (
                total_tokens + token_estimate
                > max_tokens
            ):
                break

            # Enforce per-URL cap
            url_key = metadata.get("url", "").lower().split("?")[0]  # strip query params
            if url_counts.get(url_key, 0) >= max_chunks_per_url:
                continue
            url_counts[url_key] = url_counts.get(url_key, 0) + 1

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

            retrieved_chunks.append(

                RetrievedChunk(

                    company=metadata["company"],

                    product_area=metadata["product_area"],

                    title=metadata["title"],

                    section=metadata["section"],

                    url=metadata["url"],

                    document_id=metadata.get(
                        "doc_id",
                        "",
                    ),

                    chunk_id=result["id"],

                    score=max(
                        result.get("vector_score", 0.0),
                        result.get("bm25_score", 0.0),
                    ),

                    text=result["text"],
                )
            )
            

            total_tokens += token_estimate

        return {
            "context": context,
            "retrieved_chunks": retrieved_chunks,
            "sources": sources,
            "num_chunks": len(context),
            "token_estimate": total_tokens,
        }

