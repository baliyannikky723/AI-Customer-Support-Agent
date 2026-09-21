import os
import sys
import unittest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pii_sanitizer import PIISanitizer
from src.preprocessing.language_detector import LanguageDetector
from scripts.build_amazonhelp_conversations import build_conversations
from scripts.create_amazonhelp_sample import create_dev_sample


class TestPhase2Pipeline(unittest.TestCase):
    """Unit tests for Phase 2 data cleaning, PII sanitization, and conversation reconstruction."""

    def setUp(self):
        self.sanitizer = PIISanitizer()
        self.lang_detector = LanguageDetector()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    # 1. PII Sanitization Tests
    def test_pii_sanitization_order_ids(self):
        text = "Where is my order 112-3456789-1234567 and D01-9876543-1234567?"
        sanitized = self.sanitizer.sanitize(text)
        self.assertNotIn("112-3456789-1234567", sanitized)
        self.assertNotIn("D01-9876543-1234567", sanitized)
        self.assertIn("[ORDER_ID]", sanitized)

    def test_pii_sanitization_emails_and_phones(self):
        text = "Contact me at customer.test@gmail.com or +1-800-555-0199 please."
        sanitized = self.sanitizer.sanitize(text)
        self.assertNotIn("customer.test@gmail.com", sanitized)
        self.assertNotIn("+1-800-555-0199", sanitized)
        self.assertIn("[EMAIL]", sanitized)
        self.assertIn("[PHONE]", sanitized)

    def test_pii_sanitization_urls_handles_and_signatures(self):
        text = "@105834 Check your delivery at https://amzn.to/track?id=xyz123 ^SN"
        sanitized = self.sanitizer.sanitize(text)
        self.assertNotIn("@105834", sanitized)
        self.assertNotIn("https://amzn.to/track?id=xyz123", sanitized)
        self.assertNotIn("^SN", sanitized)
        self.assertIn("[CUSTOMER]", sanitized)
        self.assertIn("[URL]", sanitized)

    def test_pii_negative_preservation(self):
        text = "I ordered an Echo Dot 3rd Gen on Oct 11 for $29.99 and it was damaged."
        sanitized = self.sanitizer.sanitize(text)
        self.assertEqual(sanitized, text)

    # 2. Language Detection Tests
    def test_language_detection(self):
        en_text = "My package has not arrived and tracking is not updating."
        es_text = "Hola, mi pedido no ha llegado y necesito ayuda con el reembolso."
        ja_text = "注文した商品が届きません。確認をお願いします。"

        self.assertEqual(self.lang_detector.detect(en_text)[0], "en")
        self.assertEqual(self.lang_detector.detect(es_text)[0], "es")
        self.assertEqual(self.lang_detector.detect(ja_text)[0], "ja")
        self.assertTrue(self.lang_detector.is_english(en_text))
        self.assertFalse(self.lang_detector.is_english(es_text))

    # 3. Conversation Reconstruction Tests
    def test_conversation_reconstruction_on_fixture(self):
        # Create a mini fixture with 2 conversations
        fixture_records = [
            # Conv 1 (2 turns)
            {"tweet_id": 201, "author_id": "cust_1", "inbound": True, "created_at": "Wed Oct 11 10:00:00 +0000 2017", "text": "My order 111-2222222-3333333 is late", "response_tweet_id": "202", "in_response_to_tweet_id": np.nan},
            {"tweet_id": 202, "author_id": "AmazonHelp", "inbound": False, "created_at": "Wed Oct 11 10:05:00 +0000 2017", "text": "@cust_1 We apologize! Track here https://amzn.to/123 ^MM", "response_tweet_id": np.nan, "in_response_to_tweet_id": 201.0},
            
            # Conv 2 (4 turns)
            {"tweet_id": 301, "author_id": "cust_2", "inbound": True, "created_at": "Wed Oct 11 11:00:00 +0000 2017", "text": "Can I return a opened item?", "response_tweet_id": "302", "in_response_to_tweet_id": np.nan},
            {"tweet_id": 302, "author_id": "AmazonHelp", "inbound": False, "created_at": "Wed Oct 11 11:05:00 +0000 2017", "text": "@cust_2 Yes! Which category? ^RG", "response_tweet_id": "303", "in_response_to_tweet_id": 301.0},
            {"tweet_id": 303, "author_id": "cust_2", "inbound": True, "created_at": "Wed Oct 11 11:10:00 +0000 2017", "text": "It is electronics.", "response_tweet_id": "304", "in_response_to_tweet_id": 302.0},
            {"tweet_id": 304, "author_id": "AmazonHelp", "inbound": False, "created_at": "Wed Oct 11 11:15:00 +0000 2017", "text": "@cust_2 You have 30 days. Visit [URL] ^RG", "response_tweet_id": np.nan, "in_response_to_tweet_id": 303.0},
        ]
        
        fixture_df = pd.DataFrame(fixture_records)
        raw_pq_path = self.temp_path / "fixture_raw.parquet"
        out_conv_path = self.temp_path / "fixture_convs.parquet"
        fixture_df.to_parquet(raw_pq_path, index=False)

        convs_df = build_conversations(
            raw_parquet_path=str(raw_pq_path),
            output_conversations_path=str(out_conv_path),
            language_mode="all",
            sanitize_pii=True,
        )

        self.assertEqual(len(convs_df), 2)
        self.assertTrue(all(convs_df["is_complete"]))
        # Verify PII was sanitized in reconstructed query
        self.assertIn("[ORDER_ID]", convs_df.loc[convs_df["conversation_id"] == "201", "customer_query"].iloc[0])

    # 4. Sampling Determinism Test
    def test_sampling_determinism(self):
        fixture_records = []
        for i in range(50):
            fixture_records.append({
                "conversation_id": str(1000 + i),
                "num_turns": 2 if i % 2 == 0 else 4,
                "is_complete": True,
                "language": "en",
                "customer_query": f"Inquiry number {i} regarding order delivery status",
                "amazon_response": f"Response number {i} with resolution details",
                "resolution_text": f"Resolution {i}",
                "start_time": "Wed Oct 11 00:00:00 +0000 2017",
                "end_time": "Wed Oct 11 00:10:00 +0000 2017",
                "message_count": 2,
                "customer_message_count": 1,
                "support_message_count": 1,
                "has_customer_message": True,
                "has_support_response": True,
                "language_confidence": 0.9,
                "customer_query_raw": f"Inquiry {i}",
                "amazon_response_raw": f"Response {i}",
            })
        
        df = pd.DataFrame(fixture_records)
        in_path = self.temp_path / "convs.parquet"
        out_1 = self.temp_path / "sample_1.parquet"
        out_2 = self.temp_path / "sample_2.parquet"
        df.to_parquet(in_path, index=False)

        sample_1 = create_dev_sample(str(in_path), str(out_1), sample_size=10, random_seed=42)
        sample_2 = create_dev_sample(str(in_path), str(out_2), sample_size=10, random_seed=42)

        self.assertEqual(len(sample_1), 10)
        self.assertEqual(len(sample_2), 10)
        pd.testing.assert_frame_equal(sample_1, sample_2)

    # 5. Raw Data Integrity
    def test_raw_data_not_mutated(self):
        raw_csv = Path("data/raw/twcs.csv")
        if raw_csv.exists():
            # Check file size matches original ~492.58 MB
            size_mb = raw_csv.stat().st_size / (1024 * 1024)
            self.assertAlmostEqual(size_mb, 492.58, delta=1.0)


if __name__ == "__main__":
    unittest.main()
