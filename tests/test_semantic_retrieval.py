"""Unit tests for Phase 6 Semantic Historical Retrieval and Vector Search Foundation."""

import os
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.retrieval.semantic_embedder import SemanticEmbedder
from src.retrieval.faiss_retriever import FAISSRetriever
from src.retrieval.retrieval_schema import RetrievalResult
from src.preprocessing.pii_sanitizer import PIISanitizer


class TestSemanticRetrieval(unittest.TestCase):
    """Test suite for SemanticEmbedder and FAISSRetriever modules."""

    @classmethod
    def setUpClass(cls):
        """Initialize embedder once for test suite to optimize runtime."""
        cls.embedder = SemanticEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            normalize_embeddings=True,
            batch_size=16,
        )

    def test_embedder_dimension_and_shape(self):
        """Verify embedding dimension is 384 and batch encoding shape is correct."""
        self.assertEqual(self.embedder.dimension, 384)

        texts = ["Where is my package?", "I want a refund for my damaged item."]
        embeddings = self.embedder.encode(texts)
        self.assertEqual(embeddings.shape, (2, 384))
        self.assertEqual(embeddings.dtype, np.float32)

    def test_embedding_l2_normalization(self):
        """Verify encoded vectors have unit L2 norm (||v||_2 = 1.0)."""
        texts = ["Late delivery complaint", "Return pickup request"]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        
        norms = np.linalg.norm(embeddings, axis=1)
        for norm in norms:
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_empty_query_handling(self):
        """Verify embedder handles empty inputs gracefully."""
        empty_emb = self.embedder.encode([])
        self.assertEqual(empty_emb.shape, (0, 384))

    def test_faiss_retriever_build_and_top_k(self):
        """Verify FAISS retriever builds index and retrieves top-k in descending order."""
        corpus = pd.DataFrame([
            {
                "conversation_id": "conv_101",
                "customer_query": "My order is delayed and has not arrived yet",
                "amazon_response": "We apologize for the delay. Please check your tracking link: [URL]",
                "start_time": "2017-10-01 10:00:00",
                "intent": "late_delivery_complaint",
            },
            {
                "conversation_id": "conv_102",
                "customer_query": "The product arrived completely broken and shattered in the box",
                "amazon_response": "We are sorry to hear that. You can request a replacement in Your Orders.",
                "start_time": "2017-10-02 11:00:00",
                "intent": "damaged_defective_or_wrong_item",
            },
            {
                "conversation_id": "conv_103",
                "customer_query": "I would like to cancel my Prime membership renewal",
                "amazon_response": "You can manage or cancel your Prime subscription under Manage Prime Membership.",
                "start_time": "2017-10-03 12:00:00",
                "intent": "prime_membership_and_digital",
            },
        ])

        retriever = FAISSRetriever(embedder=self.embedder)
        retriever.build_index(corpus)

        self.assertEqual(retriever.size, 3)
        self.assertTrue(retriever.is_indexed)

        # Query semantic match
        results = retriever.retrieve("My package is late, where is it?", top_k=2)
        self.assertEqual(len(results), 2)
        
        # Check top-1 is the delayed package case
        self.assertEqual(results[0].conversation_id, "conv_101")
        self.assertGreater(results[0].similarity_score, results[1].similarity_score)
        self.assertGreater(results[0].similarity_score, 0.5)

    def test_pii_sanitization_in_retriever(self):
        """Verify retriever sanitizes sensitive customer handles and order numbers from output."""
        corpus = pd.DataFrame([
            {
                "conversation_id": "conv_201",
                "customer_query": "Help with order 112-9876543-1234567",
                "amazon_response": "@123456 Please check order 112-9876543-1234567 at https://amazon.com/track ^AB",
                "start_time": "2017-10-01 10:00:00",
                "intent": "delivery_status_tracking",
            }
        ])

        retriever = FAISSRetriever(embedder=self.embedder)
        retriever.build_index(corpus)

        results = retriever.retrieve("Track order status", top_k=1)
        self.assertEqual(len(results), 1)

        hist_reply = results[0].historical_reply
        self.assertNotIn("@123456", hist_reply)
        self.assertNotIn("112-9876543-1234567", hist_reply)
        self.assertNotIn("^AB", hist_reply)
        self.assertIn("[CUSTOMER]", hist_reply)
        self.assertIn("[ORDER_ID]", hist_reply)

    def test_serialization_and_deserialization(self):
        """Verify FAISS index and metadata can be saved and reloaded accurately."""
        corpus = pd.DataFrame([
            {
                "conversation_id": "conv_301",
                "customer_query": "Refund status inquiry for returned shoes",
                "amazon_response": "Refunds usually take 3-5 business days to appear.",
                "start_time": "2017-10-01 10:00:00",
                "intent": "refund_status_and_request",
            }
        ])

        with tempfile.TemporaryDirectory() as tmp_dir:
            idx_file = Path(tmp_dir) / "test_index.bin"
            meta_file = Path(tmp_dir) / "test_meta.parquet"

            retriever = FAISSRetriever(embedder=self.embedder)
            retriever.build_index(corpus)
            retriever.save(idx_file, meta_file)

            # Reload
            loaded_retriever = FAISSRetriever(embedder=self.embedder)
            loaded_retriever.load(idx_file, meta_file)

            self.assertEqual(loaded_retriever.size, 1)
            res = loaded_retriever.retrieve("When will I get my refund?", top_k=1)
            self.assertEqual(res[0].conversation_id, "conv_301")
            self.assertGreater(res[0].similarity_score, 0.5)

    def test_zero_leakage_in_processed_index(self):
        """Verify the production index contains zero conversation IDs from the golden set."""
        prod_meta_path = Path("data/processed/resolution_metadata.parquet")
        golden_inputs_path = Path("data/golden/golden_inputs.jsonl")

        if prod_meta_path.exists() and golden_inputs_path.exists():
            df_meta = pd.read_parquet(prod_meta_path)
            indexed_ids = set(df_meta["conversation_id"].astype(str))

            golden_inputs = [json.loads(line) for line in golden_inputs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            golden_ids = {str(item["conversation_id"]) for item in golden_inputs}

            overlap = indexed_ids.intersection(golden_ids)
            self.assertEqual(len(overlap), 0, f"Leaked conversation IDs found: {overlap}")


if __name__ == "__main__":
    unittest.main()
