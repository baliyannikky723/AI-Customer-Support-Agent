import json
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.evaluation.baselines.majority_baseline import MajorityBaseline
from src.evaluation.baselines.tfidf_intent import TFIDFIntentClassifier
from src.evaluation.baselines.tfidf_triage import TFIDFTriageClassifier
from src.evaluation.baselines.tfidf_retriever import TFIDFRetriever


class TestBaselines(unittest.TestCase):
    """Unit tests for Phase 5 Majority and Classical TF-IDF Baselines."""

    def setUp(self):
        # Synthetic development corpus for fast testing
        self.dev_data = [
            {"conversation_id": "101", "customer_query": "Where is my package? Need tracking.", "amazon_response": "Track your order here: [URL] ^SN", "num_turns": 2},
            {"conversation_id": "102", "customer_query": "Where is my delivery tracking?", "amazon_response": "Check tracking in Your Orders: [URL] ^MM", "num_turns": 2},
            {"conversation_id": "103", "customer_query": "My order is 3 days late!", "amazon_response": "We apologize for the delay. Check [URL] ^RG", "num_turns": 4},
            {"conversation_id": "104", "customer_query": "How do I return this broken item?", "amazon_response": "Visit the Returns Center: [URL] ^SN", "num_turns": 2},
            {"conversation_id": "105", "customer_query": "I was charged twice on my credit card.", "amazon_response": "Please check your bank statement. ^CR", "num_turns": 2},
            {"conversation_id": "106", "customer_query": "Cannot log into my account. Password reset failed.", "amazon_response": "Contact account specialists: [URL] ^SN", "num_turns": 2},
        ]
        self.df_dev = pd.DataFrame(self.dev_data)
        self.intents = [
            "delivery_status_tracking",
            "delivery_status_tracking",
            "late_delivery_complaint",
            "damaged_defective_or_wrong_item",
            "payment_and_billing_issues",
            "account_access_and_security",
        ]
        self.handlings = [
            "AUTO_HANDLE",
            "AUTO_HANDLE",
            "ESCALATE",
            "AUTO_HANDLE",
            "ESCALATE",
            "ESCALATE",
        ]

    def test_majority_baseline(self):
        """Test MajorityBaseline fitting and prediction."""
        maj = MajorityBaseline().fit(self.intents, self.handlings)
        self.assertEqual(maj.majority_intent, "delivery_status_tracking")
        self.assertEqual(maj.majority_handling, "AUTO_HANDLE")

        preds = maj.predict_intent(["Sample text 1", "Sample text 2"])
        self.assertEqual(preds, ["delivery_status_tracking", "delivery_status_tracking"])

        handling_preds = maj.predict_handling(["Sample text 1", "Sample text 2"])
        self.assertEqual(handling_preds, ["AUTO_HANDLE", "AUTO_HANDLE"])

    def test_tfidf_intent_classifier(self):
        """Test TFIDFIntentClassifier fit and prediction."""
        clf = TFIDFIntentClassifier(max_features=500, random_state=42)
        X = self.df_dev["customer_query"].tolist()
        clf.fit(X, self.intents)

        test_queries = ["Where is my tracking number?", "I was charged twice on my card"]
        preds = clf.predict(test_queries)
        self.assertEqual(len(preds), 2)
        self.assertEqual(preds[0], "delivery_status_tracking")
        self.assertEqual(preds[1], "payment_and_billing_issues")

        probs = clf.predict_proba(test_queries)
        self.assertEqual(probs.shape[0], 2)
        self.assertEqual(probs.shape[1], len(set(self.intents)))

    def test_tfidf_triage_classifier(self):
        """Test TFIDFTriageClassifier binary prediction."""
        clf = TFIDFTriageClassifier(max_features=500, random_state=42)
        X = self.df_dev["customer_query"].tolist()
        clf.fit(X, self.handlings)

        test_queries = ["Where is my tracking number?", "Cannot log in, password failed"]
        preds = clf.predict(test_queries)
        self.assertEqual(len(preds), 2)
        self.assertIn(preds[0], {"AUTO_HANDLE", "ESCALATE"})
        self.assertIn(preds[1], {"AUTO_HANDLE", "ESCALATE"})

    def test_tfidf_retriever(self):
        """Test TFIDFRetriever index building and retrieval similarity."""
        retriever = TFIDFRetriever(max_features=500)
        retriever.build_index(self.df_dev)

        res = retriever.retrieve("Where is my package tracking?", top_k=1)
        self.assertEqual(len(res), 1)
        self.assertIn("retrieved_case_id", res[0])
        self.assertIn("similarity_score", res[0])
        self.assertGreater(res[0]["similarity_score"], 0.3)
        self.assertIn("[URL]", res[0]["retrieved_historical_response"])
        # Verify employee sign-off ^SN was stripped
        self.assertNotIn("^SN", res[0]["retrieved_historical_response"])

    def test_empty_and_unknown_inputs(self):
        """Verify classifiers and retriever handle empty/short inputs gracefully."""
        clf = TFIDFIntentClassifier(max_features=500).fit(self.df_dev["customer_query"].tolist(), self.intents)
        preds = clf.predict(["", " ", "???"])
        self.assertEqual(len(preds), 3)

        retriever = TFIDFRetriever(max_features=500).build_index(self.df_dev)
        res = retriever.retrieve("")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["similarity_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
