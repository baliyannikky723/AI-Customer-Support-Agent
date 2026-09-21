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

from scripts.create_dataset_sample import create_representative_sample
from scripts.explore_dataset import analyze_dataset


class TestDatasetAnalysis(unittest.TestCase):
    """Unit tests for Phase 1 dataset analysis and sampling utilities using mock fixtures."""

    def setUp(self):
        """Create a lightweight synthetic dataset fixture for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Synthetic mock dataset with 20 rows
        base_records = [
            # Amazon conversation 1
            {"tweet_id": 101, "author_id": "cust_1", "inbound": True, "created_at": "Wed Oct 11 00:00:00 +0000 2017", "text": "@AmazonHelp my order is late", "response_tweet_id": "102", "in_response_to_tweet_id": np.nan},
            {"tweet_id": 102, "author_id": "AmazonHelp", "inbound": False, "created_at": "Wed Oct 11 00:05:00 +0000 2017", "text": "@cust_1 We are sorry! Track here https://amzn.to/123 ^SN", "response_tweet_id": np.nan, "in_response_to_tweet_id": 101.0},
            
            # Apple conversation 1
            {"tweet_id": 103, "author_id": "cust_2", "inbound": True, "created_at": "Wed Oct 11 00:10:00 +0000 2017", "text": "@AppleSupport battery drains fast", "response_tweet_id": "104", "in_response_to_tweet_id": np.nan},
            {"tweet_id": 104, "author_id": "AppleSupport", "inbound": False, "created_at": "Wed Oct 11 00:15:00 +0000 2017", "text": "@cust_2 Check settings: https://apple.co/xyz", "response_tweet_id": np.nan, "in_response_to_tweet_id": 103.0},

            # Uber conversation 1
            {"tweet_id": 105, "author_id": "cust_3", "inbound": True, "created_at": "Wed Oct 11 00:20:00 +0000 2017", "text": "@Uber_Support left my wallet in cab", "response_tweet_id": "106", "in_response_to_tweet_id": np.nan},
            {"tweet_id": 106, "author_id": "Uber_Support", "inbound": False, "created_at": "Wed Oct 11 00:25:00 +0000 2017", "text": "@cust_3 Please DM us your ride details.", "response_tweet_id": np.nan, "in_response_to_tweet_id": 105.0},
        ]
        
        mock_records = []
        for i in range(10):
            for rec in base_records:
                new_rec = dict(rec)
                new_rec["tweet_id"] = int(1000 + len(mock_records))
                mock_records.append(new_rec)

        self.mock_df = pd.DataFrame(mock_records)
        self.mock_csv_path = self.temp_path / "mock_twcs.csv"
        self.mock_df.to_csv(self.mock_csv_path, index=False)

    def tearDown(self):
        """Clean up temporary fixtures."""
        self.temp_dir.cleanup()

    def test_invalid_path_raises_filenotfound(self):
        """Ensure analyzer raises FileNotFoundError when given a non-existent path."""
        with self.assertRaises(FileNotFoundError):
            analyze_dataset(raw_data_path="data/raw/non_existent_file.csv")

        with self.assertRaises(FileNotFoundError):
            create_representative_sample(raw_data_path="data/raw/non_existent_file.csv")

    def test_sample_generation_and_determinism(self):
        """Ensure sample generation produces expected size and is reproducible with fixed seed."""
        output_sample_1 = self.temp_path / "sample_1.csv"
        output_sample_2 = self.temp_path / "sample_2.csv"

        df_1 = create_representative_sample(
            raw_data_path=str(self.mock_csv_path),
            output_sample_path=str(output_sample_1),
            sample_size=10,
            random_seed=42,
        )

        df_2 = create_representative_sample(
            raw_data_path=str(self.mock_csv_path),
            output_sample_path=str(output_sample_2),
            sample_size=10,
            random_seed=42,
        )

        self.assertEqual(len(df_1), 10)
        self.assertEqual(len(df_2), 10)
        # Verify determinism
        pd.testing.assert_frame_equal(df_1, df_2)

    def test_schema_and_brand_analysis_on_fixture(self):
        """Ensure analyze_dataset extracts correct metrics and brand counts from fixture."""
        output_brand_stats = self.temp_path / "brand_stats.csv"
        summary = analyze_dataset(
            raw_data_path=str(self.mock_csv_path),
            output_brand_stats_path=str(output_brand_stats),
            chunk_size=50,
        )

        self.assertEqual(summary["total_rows"], 60)
        top_brand_names = [b["brand"] for b in summary["top_10_brands"]]
        self.assertIn("AmazonHelp", top_brand_names)
        self.assertTrue(output_brand_stats.exists())

        stats_df = pd.read_csv(output_brand_stats)
        self.assertTrue("brand" in stats_df.columns)
        self.assertTrue("usable_conversations" in stats_df.columns)
        self.assertGreater(len(stats_df), 0)


if __name__ == "__main__":
    unittest.main()
