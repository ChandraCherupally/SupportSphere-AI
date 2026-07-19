"""
Automated unit tests for the SupportSphere AI Retrieval pipeline correctness.
Verifies BM25, Vector, and Hybrid retrieval modes, deduplication, URL capping, and Top-K constraints.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import src.config as src_config
from src.retrieval.hybrid_search import HybridSearch


class TestRetrievalPipeline(unittest.TestCase):
    """
    Test suite for retrieval modes, deduplication, and Top-K guarantees.
    """

    def setUp(self) -> None:
        # Save config defaults to restore after tests
        self.original_final_top_k = getattr(src_config, "FINAL_TOP_K", 10)
        self.original_search_mode = getattr(src_config, "SEARCH_MODE", "hybrid")
        self.original_max_chunks = getattr(src_config, "MAX_CHUNKS_PER_URL", 2)

    def tearDown(self) -> None:
        # Restore configuration variables
        src_config.FINAL_TOP_K = self.original_final_top_k
        src_config.SEARCH_MODE = self.original_search_mode
        src_config.MAX_CHUNKS_PER_URL = self.original_max_chunks

    @patch("src.retrieval.hybrid_search.BM25Index")
    @patch("src.retrieval.hybrid_search.VectorSearch")
    def test_bm25_only_mode(self, mock_vector_cls, mock_bm25_cls) -> None:
        """
        Verify that BM25 mode executes only the BM25 backend.
        """
        mock_bm25_inst = mock_bm25_cls.return_value
        mock_vector_inst = mock_vector_cls.return_value

        mock_bm25_inst.search.return_value = [
            {"id": "bm25_1", "text": "BM25 content 1", "metadata": {"url": "http://a.com", "company": "Claude"}},
            {"id": "bm25_2", "text": "BM25 content 2", "metadata": {"url": "http://a.com", "company": "Claude"}},
        ]

        hybrid = HybridSearch(reranker="none", search_mode="bm25")
        results, trace = hybrid.search(query="test query", company="Claude")

        # BM25 should be called, Vector search should NOT be called
        mock_bm25_inst.search.assert_called_once()
        mock_vector_inst.search.assert_not_called()

        self.assertEqual(len(results), 2)
        self.assertEqual(trace["search_mode"], "bm25")
        self.assertEqual(trace["bm25_initial_count"], 2)
        self.assertEqual(trace["vector_initial_count"], 0)

    @patch("src.retrieval.hybrid_search.BM25Index")
    @patch("src.retrieval.hybrid_search.VectorSearch")
    def test_vector_only_mode(self, mock_vector_cls, mock_bm25_cls) -> None:
        """
        Verify that Vector mode executes only the Vector search backend.
        """
        mock_bm25_inst = mock_bm25_cls.return_value
        mock_vector_inst = mock_vector_cls.return_value

        mock_vector_inst.search.return_value = [
            {"id": "vec_1", "text": "Vector content 1", "metadata": {"url": "http://b.com", "company": "Claude"}},
        ]

        hybrid = HybridSearch(reranker="none", search_mode="vector")
        results, trace = hybrid.search(query="test query", company="Claude")

        # Vector search should be called, BM25 search should NOT be called
        mock_vector_inst.search.assert_called_once()
        mock_bm25_inst.search.assert_not_called()

        self.assertEqual(len(results), 1)
        self.assertEqual(trace["search_mode"], "vector")
        self.assertEqual(trace["vector_initial_count"], 1)
        self.assertEqual(trace["bm25_initial_count"], 0)

    @patch("src.retrieval.hybrid_search.BM25Index")
    @patch("src.retrieval.hybrid_search.VectorSearch")
    def test_hybrid_mode(self, mock_vector_cls, mock_bm25_cls) -> None:
        """
        Verify that Hybrid mode executes both backends and fuses them.
        """
        mock_bm25_inst = mock_bm25_cls.return_value
        mock_vector_inst = mock_vector_cls.return_value

        mock_bm25_inst.search.return_value = [
            {"id": "c1", "text": "shared content", "metadata": {"url": "http://c.com", "company": "Claude"}},
        ]
        mock_vector_inst.search.return_value = [
            {"id": "c1", "text": "shared content", "metadata": {"url": "http://c.com", "company": "Claude"}},
            {"id": "c2", "text": "other content", "metadata": {"url": "http://d.com", "company": "Claude"}},
        ]

        hybrid = HybridSearch(reranker="none", search_mode="hybrid")
        results, trace = hybrid.search(query="test query", company="Claude")

        # Both should be called
        mock_bm25_inst.search.assert_called_once()
        mock_vector_inst.search.assert_called_once()

        self.assertEqual(trace["search_mode"], "hybrid")
        self.assertEqual(trace["bm25_initial_count"], 1)
        self.assertEqual(trace["vector_initial_count"], 2)
        # Deduplication merges shared "c1" chunk
        self.assertEqual(trace["unique_count"], 2)

    @patch("src.retrieval.hybrid_search.BM25Index")
    @patch("src.retrieval.hybrid_search.VectorSearch")
    def test_url_cap_deduplication_loop(self, mock_vector_cls, mock_bm25_cls) -> None:
        """
        Verify that if deduplication reduces chunk count below Top-K, additional candidates are retrieved.
        """
        mock_bm25_inst = mock_bm25_cls.return_value
        mock_vector_inst = mock_vector_cls.return_value

        # Configure Top-K limit
        src_config.FINAL_TOP_K = 3
        src_config.MAX_CHUNKS_PER_URL = 1  # Strictest URL cap

        # Attempt 1 returns candidates from same URL: http://dup.com
        # Attempt 2 (limit scaled) returns additional candidates from http://other.com
        def search_side_effect(query, top_k, company=None, filters=None):
            if top_k <= 10:
                # First attempt candidate list
                return [
                    {"id": "1", "text": "text 1", "metadata": {"url": "http://dup.com", "company": "Claude"}},
                    {"id": "2", "text": "text 2", "metadata": {"url": "http://dup.com", "company": "Claude"}},
                    {"id": "3", "text": "text 3", "metadata": {"url": "http://dup.com", "company": "Claude"}},
                ]
            else:
                # Second scaled attempt candidate list
                return [
                    {"id": "1", "text": "text 1", "metadata": {"url": "http://dup.com", "company": "Claude"}},
                    {"id": "2", "text": "text 2", "metadata": {"url": "http://dup.com", "company": "Claude"}},
                    {"id": "3", "text": "text 3", "metadata": {"url": "http://dup.com", "company": "Claude"}},
                    {"id": "4", "text": "text 4", "metadata": {"url": "http://other.com", "company": "Claude"}},
                    {"id": "5", "text": "text 5", "metadata": {"url": "http://third.com", "company": "Claude"}},
                ]

        mock_bm25_inst.search.side_effect = search_side_effect

        hybrid = HybridSearch(reranker="none", search_mode="bm25")
        results, trace = hybrid.search(query="test query", company="Claude")

        # Results should respect Top-K = 3 by pulling unique candidates across scaled queries
        self.assertEqual(len(results), 3)
        self.assertEqual(trace["final_returned_count"], 3)

        # Chunks returned should have unique URLs due to the cap
        urls = [r["metadata"]["url"] for r in results]
        self.assertEqual(len(set(urls)), 3)
