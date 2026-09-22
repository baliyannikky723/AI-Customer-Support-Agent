"""Unit tests for Phase 7 Intent-Aware Retrieval, Reranking, and Pipeline Integration."""

import json
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src.evaluation.baselines.tfidf_intent import TFIDFIntentClassifier
from src.retrieval.semantic_embedder import SemanticEmbedder
from src.retrieval.faiss_retriever import FAISSRetriever
from src.retrieval.intent_aware_reranker import IntentAwareReranker
from src.retrieval.retrieval_schema import RetrievalResult, RerankedResult
from src.pipeline.intent_aware_retrieval import IntentAwareRetrievalPipeline


class TestIntentAwareRetrieval(unittest.TestCase):
    """Test suite for Intent Classifier, Intent-Aware Reranker, and Pipeline Integration."""

    @classmethod
    def setUpClass(cls):
        """Fit classifier and build retriever mock once for test suite efficiency."""
        # 1. Train lightweight TF-IDF classifier on mock cases
        cls.train_corpus = pd.DataFrame([
            {"customer_query": "where is my tracking number for package", "intent": "delivery_status_tracking"},
            {"customer_query": "package is late and delayed", "intent": "late_delivery_complaint"},
            {"customer_query": "item arrived broken cracked shattered", "intent": "damaged_defective_or_wrong_item"},
            {"customer_query": "return pickup courier not arrived", "intent": "return_and_pickup_inquiry"},
            {"customer_query": "refund not credited to my bank account", "intent": "refund_status_and_request"},
            {"customer_query": "charged twice double charge on credit card", "intent": "payment_and_billing_issues"},
            {"customer_query": "cancel my prime annual subscription", "intent": "prime_membership_and_digital"},
            {"customer_query": "locked account need password otp reset", "intent": "account_access_and_security"},
        ])

        cls.classifier = TFIDFIntentClassifier(max_features=500, min_df=1, random_state=42)
        cls.classifier.fit(cls.train_corpus["customer_query"].tolist(), cls.train_corpus["intent"].tolist())

        # 2. Retriever mock
        cls.embedder = SemanticEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            normalize_embeddings=True,
            batch_size=8,
        )
        cls.retriever = FAISSRetriever(embedder=cls.embedder)
        cls.retriever.build_index(
            corpus_df=cls.train_corpus,
            query_col="customer_query",
            reply_col="customer_query",  # Use query as placeholder reply for mock
            intent_col="intent",
        )

    def test_classifier_predict_detailed(self):
        """Verify classifier predict_detailed outputs probabilities, confidence, and top3."""
        queries = ["Where is my package tracking?", "I want a refund for late item"]
        detailed = self.classifier.predict_detailed(queries)

        self.assertEqual(len(detailed), 2)
        for d in detailed:
            self.assertIn("predicted_intent", d)
            self.assertIn("confidence", d)
            self.assertIn("probabilities", d)
            self.assertIn("top3", d)
            self.assertIsInstance(d["confidence"], float)
            self.assertGreaterEqual(d["confidence"], 0.0)
            self.assertLessEqual(d["confidence"], 1.0)
            self.assertEqual(len(d["top3"]), 3)
            # Verify top3 is sorted descending
            self.assertGreaterEqual(d["top3"][0]["probability"], d["top3"][1]["probability"])
            self.assertGreaterEqual(d["top3"][1]["probability"], d["top3"][2]["probability"])

    def test_reranker_soft_bonus_calculation(self):
        """Verify soft reranking adds intent bonus when historical intent matches predicted."""
        candidates = [
            RetrievalResult(
                conversation_id="conv_1",
                customer_message="Item was broken",
                historical_reply="We will send a replacement.",
                similarity_score=0.75,
                timestamp="2017-10-01",
                intent="damaged_defective_or_wrong_item",
                rank=1,
            ),
            RetrievalResult(
                conversation_id="conv_2",
                customer_message="When will refund arrive",
                historical_reply="Refund takes 3-5 days.",
                similarity_score=0.70,
                timestamp="2017-10-01",
                intent="refund_status_and_request",
                rank=2,
            ),
        ]

        reranker = IntentAwareReranker(intent_match_bonus=0.10, min_confidence=0.50, mode="soft_rerank")
        
        # Test 1: Predicted intent matches rank 2 candidate (refund)
        results = reranker.rerank(candidates, predicted_intent="refund_status_and_request", confidence=0.85, top_k=2)

        self.assertEqual(len(results), 2)
        # Rank 2 candidate (refund) gets 0.70 + 0.10 = 0.80, promoting it to rank 1 over rank 1 candidate (0.75)
        self.assertEqual(results[0].conversation_id, "conv_2")
        self.assertEqual(results[0].final_rerank_score, 0.80)
        self.assertTrue(results[0].intent_match)
        self.assertEqual(results[0].intent_bonus, 0.10)
        self.assertEqual(results[0].reranked_rank, 1)

        # Rank 1 candidate (damaged) gets no bonus (0.75 + 0.0 = 0.75)
        self.assertEqual(results[1].conversation_id, "conv_1")
        self.assertEqual(results[1].final_rerank_score, 0.75)
        self.assertFalse(results[1].intent_match)

    def test_reranker_low_confidence_fallback(self):
        """Verify reranker falls back to raw similarity order when confidence < min_confidence."""
        candidates = [
            RetrievalResult(
                conversation_id="conv_1",
                customer_message="Item was broken",
                historical_reply="We will replace it.",
                similarity_score=0.75,
                timestamp="2017-10-01",
                intent="damaged_defective_or_wrong_item",
                rank=1,
            ),
            RetrievalResult(
                conversation_id="conv_2",
                customer_message="Refund question",
                historical_reply="Refund takes 3-5 days.",
                similarity_score=0.70,
                timestamp="2017-10-01",
                intent="refund_status_and_request",
                rank=2,
            ),
        ]

        # Reranker threshold 0.60; query confidence 0.40
        reranker = IntentAwareReranker(intent_match_bonus=0.10, min_confidence=0.60, mode="soft_rerank")
        results = reranker.rerank(candidates, predicted_intent="refund_status_and_request", confidence=0.40, top_k=2)

        self.assertEqual(len(results), 2)
        # Fallback used: conv_1 remains rank 1 based on raw similarity
        self.assertEqual(results[0].conversation_id, "conv_1")
        self.assertTrue(results[0].fallback_used)
        self.assertEqual(results[0].intent_bonus, 0.0)
        self.assertEqual(results[0].final_rerank_score, 0.75)

    def test_reranker_hard_filter_mode(self):
        """Verify hard filter mode restricts candidates strictly to matching intent."""
        candidates = [
            RetrievalResult(
                conversation_id="conv_1",
                customer_message="Item was broken",
                historical_reply="Replacement sent.",
                similarity_score=0.75,
                timestamp="2017-10-01",
                intent="damaged_defective_or_wrong_item",
                rank=1,
            ),
            RetrievalResult(
                conversation_id="conv_2",
                customer_message="Refund inquiry",
                historical_reply="Refund processed.",
                similarity_score=0.70,
                timestamp="2017-10-01",
                intent="refund_status_and_request",
                rank=2,
            ),
        ]

        reranker = IntentAwareReranker(intent_match_bonus=0.10, min_confidence=0.50, mode="hard_intent_filter")
        results = reranker.rerank(candidates, predicted_intent="refund_status_and_request", confidence=0.80, top_k=2)

        # Only conv_2 matches
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].conversation_id, "conv_2")
        self.assertEqual(results[0].historical_intent, "refund_status_and_request")

    def test_pipeline_end_to_end_query(self):
        """Verify IntentAwareRetrievalPipeline executes end-to-end and populates full bundle."""
        pipeline = IntentAwareRetrievalPipeline(
            classifier=self.classifier,
            retriever=self.retriever,
            reranker=IntentAwareReranker(intent_match_bonus=0.10, min_confidence=0.50),
        )

        bundle = pipeline.process_query(
            query_text="Where is my tracking number?",
            query_id="test_q1",
            candidate_k=4,
            top_k=2,
            gold_intent="delivery_status_tracking",
        )

        self.assertEqual(bundle.query_id, "test_q1")
        self.assertEqual(bundle.gold_intent, "delivery_status_tracking")
        self.assertEqual(len(bundle.results), 2)
        self.assertIn("predicted_intent", bundle.intent_prediction.model_dump())
        self.assertGreaterEqual(bundle.top_1_rerank_score, bundle.top_1_semantic_similarity)


if __name__ == "__main__":
    unittest.main()
