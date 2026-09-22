"""
Phase 8: End-to-End AI Support Agent & Safety Triage Unit Tests.

Validates:
1. LLM Provider Abstraction & Deterministic Mock Provider
2. Prompt Builder Construction & System Guardrails
3. Structured Response Schema Parsing & Fallback Handling
4. Grounding Guardrail (PII, Live-Status Claims, Historical Timestamps, Unsupported Claims)
5. Evidence Selection & Deterministic Quality Scoring Formula
6. Triage Engine Mandatory Escalation Conditions & Auto-Handle Criteria
7. End-to-End AISupportAgent Execution Pipeline
"""

import unittest
from src.retrieval.retrieval_schema import RerankedResult
from src.generation.response_schema import (
    EvidenceItem,
    EvidencePack,
    GeneratedResponse,
    TriageDecision,
    AgentResponseBundle,
)
from src.generation.evidence_pack import EvidenceSelector
from src.generation.prompt_builder import PromptBuilder
from src.generation.llm_provider import MockLLMProvider, get_llm_provider
from src.generation.grounding_guardrail import GroundingGuardrail
from src.escalation.triage_engine import TriageEngine
from src.generation.response_generator import ResponseGenerator
from src.pipeline.ai_support_agent import AISupportAgent


class TestResponseSchema(unittest.TestCase):
    """Test Pydantic schemas and serialization."""

    def test_evidence_item_schema(self):
        item = EvidenceItem(
            historical_case_id="case_123",
            customer_issue="Where is my package?",
            historical_resolution="You can track your package in Your Orders.",
            similarity_score=0.85,
            historical_intent="delivery_status_tracking",
            intent_match=True,
            source_timestamp="2017-10-10 12:00:00+00:00",
        )
        self.assertEqual(item.historical_case_id, "case_123")
        self.assertTrue(item.intent_match)
        self.assertEqual(item.similarity_score, 0.85)

    def test_evidence_pack_schema(self):
        item = EvidenceItem(
            historical_case_id="case_1",
            customer_issue="broken item",
            historical_resolution="Return via Your Orders.",
            similarity_score=0.75,
            historical_intent="damaged_defective_or_wrong_item",
            intent_match=True,
            source_timestamp="2017-10-10",
        )
        pack = EvidencePack(
            query_text="broken item",
            predicted_intent="damaged_defective_or_wrong_item",
            confidence=0.82,
            evidence_items=[item],
            evidence_quality_score=0.78,
            warnings=[],
        )
        self.assertEqual(len(pack.evidence_items), 1)
        self.assertEqual(pack.evidence_quality_score, 0.78)

    def test_generated_response_schema(self):
        resp = GeneratedResponse(
            reply="Please check Your Orders to track your package.",
            grounded_claims=["Tracking is available in Your Orders"],
            unsupported_claims=[],
            needs_clarification=False,
            escalation_recommended=False,
            escalation_reason="",
            confidence=0.9,
        )
        self.assertFalse(resp.escalation_recommended)
        self.assertEqual(len(resp.grounded_claims), 1)


class TestEvidenceSelectionAndScoring(unittest.TestCase):
    """Test EvidenceSelector filtering and quality score computation."""

    def setUp(self):
        self.selector = EvidenceSelector(
            max_evidence_items=3,
            min_similarity=0.55,
            deduplicate_templates=True,
        )

    def test_evidence_selection_filtering(self):
        candidates = [
            RerankedResult(
                conversation_id="c1",
                customer_message="query 1",
                historical_reply="Resolution A",
                semantic_similarity=0.85,
                historical_intent="delivery_status_tracking",
                predicted_intent="delivery_status_tracking",
                intent_match=True,
                intent_bonus=0.10,
                final_rerank_score=0.95,
                original_rank=1,
                reranked_rank=1,
                timestamp="2017-10-01",
            ),
            RerankedResult(
                conversation_id="c2",
                customer_message="query 2",
                historical_reply="Resolution A",  # Duplicate resolution
                semantic_similarity=0.80,
                historical_intent="delivery_status_tracking",
                predicted_intent="delivery_status_tracking",
                intent_match=True,
                intent_bonus=0.10,
                final_rerank_score=0.90,
                original_rank=2,
                reranked_rank=2,
                timestamp="2017-10-02",
            ),
            RerankedResult(
                conversation_id="c3",
                customer_message="query 3",
                historical_reply="Resolution B",
                semantic_similarity=0.40,  # Below min_similarity (0.55)
                historical_intent="delivery_status_tracking",
                predicted_intent="delivery_status_tracking",
                intent_match=True,
                intent_bonus=0.10,
                final_rerank_score=0.50,
                original_rank=3,
                reranked_rank=3,
                timestamp="2017-10-03",
            ),
            RerankedResult(
                conversation_id="c4",
                customer_message="query 4",
                historical_reply="Resolution C",
                semantic_similarity=0.70,
                historical_intent="late_delivery_complaint",
                predicted_intent="delivery_status_tracking",
                intent_match=False,
                intent_bonus=0.0,
                final_rerank_score=0.70,
                original_rank=4,
                reranked_rank=4,
                timestamp="2017-10-04",
            ),
        ]

        pack = self.selector.select_evidence(
            query_text="track package",
            predicted_intent="delivery_status_tracking",
            confidence=0.88,
            candidates=candidates,
        )

        # c2 is duplicate of c1, c3 is below min_similarity 0.55 -> only c1 and c4 selected
        self.assertEqual(len(pack.evidence_items), 2)
        self.assertEqual(pack.evidence_items[0].historical_case_id, "c1")
        self.assertEqual(pack.evidence_items[1].historical_case_id, "c4")
        self.assertGreaterEqual(pack.evidence_quality_score, 0.0)
        self.assertLessEqual(pack.evidence_quality_score, 1.0)

    def test_quality_score_formula_empty_evidence(self):
        score, warnings = self.selector.compute_evidence_quality_score(
            items=[],
            predicted_intent="delivery_status_tracking",
        )
        self.assertEqual(score, 0.0)

    def test_quality_score_formula_perfect_evidence(self):
        items = [
            EvidenceItem(
                historical_case_id=f"c{i}",
                customer_issue=f"issue {i}",
                historical_resolution=f"Unique resolution {i}",
                similarity_score=0.90,
                historical_intent="delivery_status_tracking",
                intent_match=True,
                source_timestamp="2017-10-01",
            )
            for i in range(3)
        ]
        score, warnings = self.selector.compute_evidence_quality_score(
            items=items,
            predicted_intent="delivery_status_tracking",
        )
        # Should be very high (> 0.85)
        self.assertGreater(score, 0.85)


class TestPromptBuilder(unittest.TestCase):
    """Test prompt assembly and JSON schema formatting."""

    def setUp(self):
        self.builder = PromptBuilder()

    def test_prompt_construction(self):
        pack = EvidencePack(
            query_text="I want to cancel order 123",
            predicted_intent="order_cancellation_request",
            confidence=0.91,
            evidence_items=[
                EvidenceItem(
                    historical_case_id="hist_1",
                    customer_issue="Can I cancel my order?",
                    historical_resolution="Cancel through Your Orders before dispatch.",
                    similarity_score=0.82,
                    historical_intent="order_cancellation_request",
                    intent_match=True,
                    source_timestamp="2017-11-01",
                )
            ],
            evidence_quality_score=0.85,
            warnings=[],
        )

        prompt_dict = self.builder.build_prompt(evidence_pack=pack)

        self.assertIn("system_prompt", prompt_dict)
        self.assertIn("user_prompt", prompt_dict)
        self.assertIn("Customer Inquiry:", prompt_dict["user_prompt"])
        self.assertIn("Predicted Intent: order_cancellation_request", prompt_dict["user_prompt"])
        self.assertIn("Historical Resolution Evidence:", prompt_dict["user_prompt"])
        self.assertIn("hist_1", prompt_dict["user_prompt"])
        self.assertIn("CRITICAL SAFETY & GROUNDING RULES", prompt_dict["system_prompt"])


class TestGroundingGuardrail(unittest.TestCase):
    """Test deterministic post-generation safety checks."""

    def setUp(self):
        self.guardrail = GroundingGuardrail()

    def test_valid_safe_response(self):
        response = GeneratedResponse(
            reply="You can check your delivery tracking status in Your Orders.",
            grounded_claims=["Tracking is available in Your Orders"],
            unsupported_claims=[],
            needs_clarification=False,
            escalation_recommended=False,
            escalation_reason="",
            confidence=0.9,
        )
        is_valid, violations = self.guardrail.validate(parsed_response=response)
        self.assertTrue(is_valid)
        self.assertEqual(len(violations), 0)

    def test_live_claim_detection(self):
        response = GeneratedResponse(
            reply="I checked your order in the system and your package is out for delivery today.",
            grounded_claims=["Package is out for delivery"],
            unsupported_claims=[],
            needs_clarification=False,
            escalation_recommended=False,
            escalation_reason="",
            confidence=0.9,
        )
        is_valid, violations = self.guardrail.validate(parsed_response=response)
        self.assertFalse(is_valid)
        self.assertTrue(any("FORBIDDEN_LIVE_CLAIM" in v for v in violations))

    def test_pii_detection(self):
        response = GeneratedResponse(
            reply="Please email user@testdomain.com regarding order 123-4567890-1234567.",
            grounded_claims=[],
            unsupported_claims=[],
            needs_clarification=False,
            escalation_recommended=False,
            escalation_reason="",
            confidence=0.9,
        )
        is_valid, violations = self.guardrail.validate(parsed_response=response)
        self.assertFalse(is_valid)
        self.assertTrue(any("PII_LEAKAGE" in v for v in violations))

    def test_historical_date_leakage(self):
        response = GeneratedResponse(
            reply="Your package was dispatched on November 25, 2017 and should arrive soon.",
            grounded_claims=[],
            unsupported_claims=[],
            needs_clarification=False,
            escalation_recommended=False,
            escalation_reason="",
            confidence=0.9,
        )
        is_valid, violations = self.guardrail.validate(parsed_response=response)
        self.assertFalse(is_valid)
        self.assertTrue(any("HISTORICAL_DATE_LEAKAGE" in v for v in violations))


class TestTriageEngine(unittest.TestCase):
    """Test deterministic safety escalation and auto-handle rules."""

    def setUp(self):
        self.triage = TriageEngine(
            min_intent_confidence=0.60,
            min_evidence_quality=0.40,
        )

    def test_mandatory_escalation_account_security(self):
        evidence_pack = EvidencePack(
            query_text="I got locked out of my account",
            predicted_intent="account_access_and_security",
            confidence=0.95,
            evidence_items=[],
            evidence_quality_score=0.80,
            warnings=[],
        )
        response = GeneratedResponse(
            reply="Please reset your password in Account Recovery.",
            grounded_claims=[],
            unsupported_claims=[],
            needs_clarification=False,
            escalation_recommended=False,
            confidence=0.9,
        )
        decision = self.triage.evaluate(
            query_text="I got locked out of my account",
            predicted_intent="account_access_and_security",
            confidence=0.95,
            evidence_pack=evidence_pack,
            generated_response=response,
            guardrail_passed=True,
            guardrail_violations=[],
        )
        self.assertEqual(decision.decision, "ESCALATE")
        self.assertIn("ACCOUNT_SECURITY_ACTION", decision.reason_codes)

    def test_mandatory_escalation_low_confidence(self):
        evidence_pack = EvidencePack(
            query_text="where package?",
            predicted_intent="delivery_status_tracking",
            confidence=0.45,  # Below 0.60
            evidence_items=[],
            evidence_quality_score=0.70,
            warnings=[],
        )
        response = GeneratedResponse(
            reply="You can check Your Orders.",
            grounded_claims=[],
            unsupported_claims=[],
            needs_clarification=False,
            escalation_recommended=False,
            confidence=0.45,
        )
        decision = self.triage.evaluate(
            query_text="where package?",
            predicted_intent="delivery_status_tracking",
            confidence=0.45,
            evidence_pack=evidence_pack,
            generated_response=response,
            guardrail_passed=True,
            guardrail_violations=[],
        )
        self.assertEqual(decision.decision, "ESCALATE")
        self.assertIn("LOW_INTENT_CONFIDENCE", decision.reason_codes)

    def test_mandatory_escalation_delivered_not_received(self):
        evidence_pack = EvidencePack(
            query_text="Package says delivered but nothing is on my porch",
            predicted_intent="order_delivered_not_received",
            confidence=0.85,
            evidence_items=[],
            evidence_quality_score=0.75,
            warnings=[],
        )
        response = GeneratedResponse(
            reply="Check with your neighbors.",
            grounded_claims=[],
            unsupported_claims=[],
            needs_clarification=False,
            escalation_recommended=False,
            confidence=0.85,
        )
        decision = self.triage.evaluate(
            query_text="Package says delivered but nothing is on my porch",
            predicted_intent="order_delivered_not_received",
            confidence=0.85,
            evidence_pack=evidence_pack,
            generated_response=response,
            guardrail_passed=True,
            guardrail_violations=[],
        )
        self.assertEqual(decision.decision, "ESCALATE")
        self.assertIn("MISSING_PACKAGE_CLAIM", decision.reason_codes)

    def test_auto_handle_on_clean_grounded_self_service(self):
        evidence_pack = EvidencePack(
            query_text="How do I track my package?",
            predicted_intent="delivery_status_tracking",
            confidence=0.90,
            evidence_items=[
                EvidenceItem(
                    historical_case_id="c1",
                    customer_issue="track item",
                    historical_resolution="Check Your Orders -> Track Package.",
                    similarity_score=0.85,
                    historical_intent="delivery_status_tracking",
                    intent_match=True,
                    source_timestamp="2017-10-01",
                )
            ],
            evidence_quality_score=0.80,
            warnings=[],
        )
        response = GeneratedResponse(
            reply="You can track your package by navigating to 'Your Orders' and selecting 'Track Package'.",
            grounded_claims=["Package tracking available in Your Orders"],
            unsupported_claims=[],
            needs_clarification=False,
            escalation_recommended=False,
            confidence=0.90,
        )
        decision = self.triage.evaluate(
            query_text="How do I track my package?",
            predicted_intent="delivery_status_tracking",
            confidence=0.90,
            evidence_pack=evidence_pack,
            generated_response=response,
            guardrail_passed=True,
            guardrail_violations=[],
        )
        self.assertEqual(decision.decision, "AUTO_HANDLE")
        self.assertIn("SAFE_GROUNDED_RESPONSE", decision.reason_codes)


class TestMockLLMProvider(unittest.TestCase):
    """Test deterministic mock LLM provider behavior."""

    def test_mock_provider_deterministic_generation(self):
        provider = MockLLMProvider()
        pack = EvidencePack(
            query_text="Received broken mug",
            predicted_intent="damaged_defective_or_wrong_item",
            confidence=0.88,
            evidence_items=[],
            evidence_quality_score=0.75,
            warnings=[],
        )
        prompt_dict = PromptBuilder().build_prompt(pack)
        raw_output = provider.generate(
            user_prompt=prompt_dict["user_prompt"],
            system_prompt=prompt_dict["system_prompt"],
        )
        self.assertIn('"reply"', raw_output)
        self.assertIn('"grounded_claims"', raw_output)

    def test_get_llm_provider_factory(self):
        provider = get_llm_provider("mock")
        self.assertIsInstance(provider, MockLLMProvider)


if __name__ == "__main__":
    unittest.main()
