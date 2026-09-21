import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestIntentTaxonomy(unittest.TestCase):
    """Unit tests for Phase 3 intent taxonomy structure, completeness, and consistency."""

    def setUp(self):
        self.taxonomy_path = PROJECT_ROOT / "data" / "processed" / "intent_taxonomy.json"
        self.guidelines_path = PROJECT_ROOT / "data" / "processed" / "intent_annotation_guidelines.md"
        self.assertTrue(self.taxonomy_path.exists(), "intent_taxonomy.json not found")
        
        with open(self.taxonomy_path, "r", encoding="utf-8") as f:
            self.taxonomy = json.load(f)

    def test_top_level_schema(self):
        """Verify top level taxonomy structure."""
        self.assertIn("version", self.taxonomy)
        self.assertIn("brand", self.taxonomy)
        self.assertIn("total_intents", self.taxonomy)
        self.assertIn("intents", self.taxonomy)
        self.assertEqual(self.taxonomy["brand"], "AmazonHelp")
        self.assertEqual(self.taxonomy["total_intents"], len(self.taxonomy["intents"]))

    def test_unique_intent_names(self):
        """Verify all intent names are unique and non-empty."""
        intent_names = [item["intent_name"] for item in self.taxonomy["intents"]]
        self.assertEqual(len(intent_names), len(set(intent_names)))
        self.assertIn("other_unknown", intent_names)
        self.assertGreaterEqual(len(intent_names), 8)
        self.assertLessEqual(len(intent_names), 15)

    def test_required_intent_fields(self):
        """Verify every intent definition contains all required metadata fields."""
        required_fields = [
            "intent_name",
            "description",
            "inclusion_criteria",
            "exclusion_criteria",
            "representative_examples",
            "approximate_frequency_pct",
            "typical_historical_resolution",
            "keywords",
            "confusable_with",
            "auto_handle_suitability",
            "escalation_considerations",
        ]
        
        for intent in self.taxonomy["intents"]:
            name = intent.get("intent_name", "UNKNOWN")
            for field in required_fields:
                self.assertIn(field, intent, f"Field '{field}' missing in intent '{name}'")
                
            # Verify representative examples
            examples = intent.get("representative_examples", [])
            self.assertIsInstance(examples, list)
            self.assertGreaterEqual(len(examples), 2, f"Intent '{name}' must have at least 2 examples")
            for ex in examples:
                self.assertIsInstance(ex, str)
                self.assertGreater(len(ex.strip()), 5)

    def test_guidelines_consistency(self):
        """Verify human annotation guidelines document references all defined intents."""
        self.assertTrue(self.guidelines_path.exists(), "intent_annotation_guidelines.md not found")
        guidelines_text = self.guidelines_path.read_text(encoding="utf-8")
        
        for intent in self.taxonomy["intents"]:
            name = intent["intent_name"]
            self.assertIn(name, guidelines_text, f"Intent '{name}' not found in annotation guidelines")


if __name__ == "__main__":
    unittest.main()
