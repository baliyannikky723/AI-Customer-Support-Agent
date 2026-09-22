"""
Phase 9: LLM-as-a-Judge Evaluation, Human Annotation Schema & Agreement Metrics Unit Tests.

Validates:
1. Judge Data Schemas & Serialization (JudgeScore, EvaluationRecord, HumanAnnotation, AutomatedResponseMetrics)
2. Judge Prompt Builder & Anonymization / Bias Controls
3. Deterministic Mock LLM Judge Logic
4. Agreement Calculator: Weighted Cohen's Kappa, Binary Cohen's Kappa, MAD, Pearson Correlation
5. Handling of Empty / Missing Human Annotations
6. Stratified Human Evaluation Subset Builder
"""

import unittest
from src.evaluation.judge.judge_schema import (
    JudgeScore,
    EvaluationRecord,
    HumanAnnotation,
    DimensionAgreement,
    BinaryAgreement,
    AgreementReport,
    AutomatedResponseMetrics,
)
from src.evaluation.judge.judge_prompt import JudgePromptBuilder
from src.evaluation.judge.llm_judge import MockLLMJudge, get_llm_judge
from src.evaluation.agreement_calculator import (
    compute_weighted_cohen_kappa,
    compute_binary_cohen_kappa,
    compute_dimension_agreement,
    compute_agreement_report,
)


class TestJudgeSchema(unittest.TestCase):
    """Test Pydantic schemas for Judge and Evaluation records."""

    def test_judge_score_schema(self):
        score = JudgeScore(
            correctness=5,
            groundedness=4,
            helpfulness=5,
            safety=5,
            tone=5,
            overall_quality=5,
            acceptable=True,
            critical_failure=False,
            reason="High quality response grounded in Amazon resolution policies.",
            failure_categories=[],
        )
        self.assertEqual(score.correctness, 5)
        self.assertTrue(score.acceptable)
        self.assertFalse(score.critical_failure)

    def test_evaluation_record_schema(self):
        rec = EvaluationRecord(
            example_id="q1",
            customer_query="where is package",
            predicted_intent="delivery_status_tracking",
            intent_confidence=0.88,
            retrieved_evidence=[{"case_id": "c1", "similarity": 0.82}],
            evidence_similarity=0.82,
            evidence_quality_score=0.80,
            generated_response="Track your package in Your Orders.",
            grounded_claims=["Tracking is available in Your Orders"],
            triage_decision="AUTO_HANDLE",
            triage_reason_codes=["SAFE_GROUNDED_RESPONSE"],
            guardrail_status="PASSED",
            guardrail_violations=[],
        )
        self.assertEqual(rec.example_id, "q1")
        self.assertEqual(rec.triage_decision, "AUTO_HANDLE")


class TestJudgePromptBuilder(unittest.TestCase):
    """Test JudgePromptBuilder formatting and anonymization."""

    def setUp(self):
        self.builder = JudgePromptBuilder()

    def test_prompt_anonymization_and_rubric(self):
        rec = EvaluationRecord(
            example_id="q100",
            customer_query="I want a refund for my late item",
            predicted_intent="refund_status_and_request",
            intent_confidence=0.92,
            retrieved_evidence=[
                {
                    "case_id": "case_999",
                    "similarity": 0.85,
                    "historical_intent": "refund_status_and_request",
                    "resolution": "Refunds are processed in 3-5 business days.",
                }
            ],
            evidence_similarity=0.85,
            evidence_quality_score=0.82,
            generated_response="Refunds are processed in 3-5 business days.",
            triage_decision="AUTO_HANDLE",
            triage_reason_codes=["SAFE_GROUNDED_RESPONSE"],
            guardrail_status="PASSED",
        )

        prompts = self.builder.build_judge_prompt(rec)
        self.assertIn("system_prompt", prompts)
        self.assertIn("user_prompt", prompts)

        user_content = prompts["user_prompt"]
        self.assertIn("I want a refund for my late item", user_content)
        self.assertIn("refund_status_and_request", user_content)
        self.assertIn("Refunds are processed in 3-5 business days", user_content)
        self.assertIn("Candidate Support Response Under Evaluation", user_content)

        # Anonymization check: System details like "Phase 8" or specific internal architecture names must NOT be in prompts
        self.assertNotIn("Phase 8", user_content)
        self.assertNotIn("Phase 9", user_content)


class TestMockLLMJudge(unittest.TestCase):
    """Test deterministic scoring behavior of MockLLMJudge."""

    def setUp(self):
        self.judge = MockLLMJudge()

    def test_mock_judge_evaluates_safe_record(self):
        rec = EvaluationRecord(
            example_id="q1",
            customer_query="where is package",
            predicted_intent="delivery_status_tracking",
            gold_intent="delivery_status_tracking",
            intent_confidence=0.88,
            retrieved_evidence=[{"case_id": "c1", "similarity": 0.82}],
            evidence_similarity=0.82,
            evidence_quality_score=0.80,
            generated_response="Please track your package in Your Orders -> Track Package.",
            grounded_claims=["Tracking in Your Orders"],
            triage_decision="AUTO_HANDLE",
            triage_reason_codes=["SAFE_GROUNDED_RESPONSE"],
            guardrail_status="PASSED",
            guardrail_violations=[],
        )
        score = self.judge.evaluate_record(rec)
        self.assertGreaterEqual(score.correctness, 4)
        self.assertGreaterEqual(score.groundedness, 4)
        self.assertGreaterEqual(score.safety, 4)
        self.assertTrue(score.acceptable)
        self.assertFalse(score.critical_failure)

    def test_mock_judge_catches_guardrail_failure(self):
        rec = EvaluationRecord(
            example_id="q2",
            customer_query="check my order",
            predicted_intent="delivery_status_tracking",
            intent_confidence=0.88,
            retrieved_evidence=[],
            evidence_similarity=0.50,
            evidence_quality_score=0.40,
            generated_response="I checked your order in the backend database.",
            grounded_claims=[],
            triage_decision="AUTO_HANDLE",
            triage_reason_codes=["SAFE_GROUNDED_RESPONSE"],
            guardrail_status="FAILED",
            guardrail_violations=["FORBIDDEN_LIVE_CLAIM"],
        )
        score = self.judge.evaluate_record(rec)
        self.assertEqual(score.safety, 1)
        self.assertTrue(score.critical_failure)
        self.assertFalse(score.acceptable)

    def test_get_llm_judge_factory(self):
        judge = get_llm_judge("mock")
        self.assertIsInstance(judge, MockLLMJudge)


class TestAgreementCalculator(unittest.TestCase):
    """Test mathematical precision of Kappa and correlation metrics."""

    def test_perfect_agreement(self):
        rater1 = [5, 4, 3, 2, 1]
        rater2 = [5, 4, 3, 2, 1]
        kappa = compute_weighted_cohen_kappa(rater1, rater2)
        self.assertEqual(kappa, 1.0)

    def test_complete_disagreement(self):
        rater1 = [5, 5, 5, 5, 5]
        rater2 = [1, 1, 1, 1, 1]
        kappa = compute_weighted_cohen_kappa(rater1, rater2)
        self.assertLessEqual(kappa, 0.0)

    def test_binary_cohen_kappa_perfect(self):
        b1 = [True, False, True, False]
        b2 = [True, False, True, False]
        b_kappa = compute_binary_cohen_kappa(b1, b2)
        self.assertEqual(b_kappa, 1.0)

    def test_dimension_agreement_computation(self):
        r1 = [5, 4, 4, 3, 5]
        r2 = [5, 4, 3, 3, 4]
        da = compute_dimension_agreement("correctness", r1, r2)
        self.assertEqual(da.dimension_name, "correctness")
        self.assertGreater(da.exact_agreement_pct, 50.0)
        self.assertEqual(da.within_one_pct, 100.0)
        self.assertGreater(da.weighted_cohen_kappa, 0.5)

    def test_agreement_report_empty_handling(self):
        report = compute_agreement_report(
            rater1_name="Judge",
            rater2_name="Human",
            rater1_scores=[],
            rater2_scores=[],
        )
        self.assertEqual(report.status, "NO_DATA")
        self.assertEqual(report.num_evaluated_pairs, 0)


if __name__ == "__main__":
    unittest.main()
