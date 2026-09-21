import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.preprocessing.pii_sanitizer import PIISanitizer
from scripts.validate_golden_isolation import validate_golden_isolation


class TestGoldenEvaluationSet(unittest.TestCase):
    """Unit tests for the 200-case Golden Evaluation Set integrity, schema, and isolation."""

    def setUp(self):
        self.inputs_path = PROJECT_ROOT / "data" / "golden" / "golden_inputs.jsonl"
        self.labels_path = PROJECT_ROOT / "data" / "golden" / "golden_labels.jsonl"
        self.taxonomy_path = PROJECT_ROOT / "data" / "processed" / "intent_taxonomy.json"
        self.policy_path = PROJECT_ROOT / "data" / "golden" / "handling_policy.md"

        self.assertTrue(self.inputs_path.exists(), "golden_inputs.jsonl not found")
        self.assertTrue(self.labels_path.exists(), "golden_labels.jsonl not found")
        self.assertTrue(self.taxonomy_path.exists(), "intent_taxonomy.json not found")
        self.assertTrue(self.policy_path.exists(), "handling_policy.md not found")

        self.inputs = [json.loads(line) for line in self.inputs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.labels = [json.loads(line) for line in self.labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        with open(self.taxonomy_path, "r", encoding="utf-8") as f:
            self.taxonomy = json.load(f)
        self.valid_intents = set(item["intent_name"] for item in self.taxonomy["intents"])

    def test_golden_dataset_size(self):
        """Verify the golden dataset contains exactly 200 examples."""
        self.assertEqual(len(self.inputs), 200)
        self.assertEqual(len(self.labels), 200)

    def test_inputs_and_labels_alignment(self):
        """Verify 1-to-1 matching between golden inputs and labels."""
        input_ids = [item["example_id"] for item in self.inputs]
        label_ids = [item["example_id"] for item in self.labels]
        self.assertEqual(input_ids, label_ids)
        self.assertEqual(len(input_ids), len(set(input_ids)))

        input_conv_ids = [item["conversation_id"] for item in self.inputs]
        label_conv_ids = [item["conversation_id"] for item in self.labels]
        self.assertEqual(input_conv_ids, label_conv_ids)
        self.assertEqual(len(input_conv_ids), len(set(input_conv_ids)))

    def test_valid_intent_and_handling_labels(self):
        """Verify all gold labels strictly match defined intent and triage taxonomy."""
        for label_item in self.labels:
            intent = label_item["gold_intent"]
            handling = label_item["gold_handling"]
            reason = label_item["gold_escalation_reason"]
            
            self.assertIn(intent, self.valid_intents, f"Invalid intent '{intent}' in golden labels")
            self.assertIn(handling, {"AUTO_HANDLE", "ESCALATE"}, f"Invalid handling '{handling}'")
            
            if handling == "ESCALATE":
                self.assertGreater(len(reason.strip()), 5, f"Missing escalation reason for '{intent}'")

    def test_pii_sanitization_in_golden_set(self):
        """Ensure no raw customer PII exists in golden set."""
        sanitizer = PIISanitizer()
        for item in self.inputs:
            text = item["customer_message"]
            counts = sanitizer.count_pii_entities(text)
            self.assertEqual(counts["orders"], 0, f"Unmasked order ID in: {text}")
            self.assertEqual(counts["emails"], 0, f"Unmasked email in: {text}")
            self.assertEqual(counts["phones"], 0, f"Unmasked phone in: {text}")

    def test_automated_isolation_audit(self):
        """Run complete isolation audit and verify zero data leakage."""
        result = validate_golden_isolation(
            conversations_path=str(PROJECT_ROOT / "data" / "processed" / "amazonhelp_conversations.parquet"),
            dev_sample_path=str(PROJECT_ROOT / "data" / "samples" / "amazonhelp_dev_sample.parquet"),
            intent_cases_path=str(PROJECT_ROOT / "data" / "samples" / "intent_discovery_cases.parquet"),
            golden_inputs_path=str(self.inputs_path),
            golden_labels_path=str(self.labels_path),
        )
        self.assertEqual(result["status"], "PASSED")
        self.assertEqual(result["overlap_count"], 0)


if __name__ == "__main__":
    unittest.main()
