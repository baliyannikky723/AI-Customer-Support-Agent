"""
Data Schemas for LLM-as-a-Judge Evaluation, Human Annotation, and Agreement Analysis.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class JudgeScore(BaseModel):
    """Structured rubric evaluation score from an LLM Judge."""
    correctness: int = Field(ge=1, le=5, description="1 (wrong) to 5 (fully correct)")
    groundedness: int = Field(ge=1, le=5, description="1 (fabricated) to 5 (fully grounded in evidence)")
    helpfulness: int = Field(ge=1, le=5, description="1 (useless) to 5 (directly actionable)")
    safety: int = Field(ge=1, le=5, description="1 (dangerous) to 5 (strongly safe/conservative)")
    tone: int = Field(ge=1, le=5, description="1 (inappropriate) to 5 (excellent empathetic)")
    overall_quality: int = Field(ge=1, le=5, description="1 to 5 aggregate judgment")
    acceptable: bool = Field(description="Whether the response is acceptable for customer facing delivery")
    critical_failure: bool = Field(description="Whether response has severe hallucination, PII leak, or safety breach")
    reason: str = Field(description="Concise rationale for the assigned scores")
    failure_categories: List[str] = Field(default_factory=list, description="List of identified failure categories if any")


class EvaluationRecord(BaseModel):
    """Complete structured record for evaluating a single generated support interaction."""
    example_id: str
    customer_query: str
    gold_intent: Optional[str] = None
    gold_handling: Optional[str] = None
    predicted_intent: str
    intent_confidence: float
    retrieved_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_similarity: float
    evidence_quality_score: float = 0.0
    generated_response: str
    grounded_claims: List[str] = Field(default_factory=list)
    triage_decision: str
    triage_reason_codes: List[str] = Field(default_factory=list)
    guardrail_status: str
    guardrail_violations: List[str] = Field(default_factory=list)
    generation_provider: str = "mock"
    generation_model: str = "deterministic_mock"
    evaluation_timestamp: str = ""
    judge_score: Optional[JudgeScore] = None


class HumanAnnotation(BaseModel):
    """Schema for human reviewer annotations."""
    example_id: str
    rater_id: str = "human_evaluator_1"
    correctness: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    helpfulness: int = Field(ge=1, le=5)
    safety: int = Field(ge=1, le=5)
    tone: int = Field(ge=1, le=5)
    overall_quality: int = Field(ge=1, le=5)
    acceptable: bool
    critical_failure: bool
    failure_categories: List[str] = Field(default_factory=list)
    rationale: Optional[str] = ""


class DimensionAgreement(BaseModel):
    """Agreement statistics for a single evaluation rubric dimension."""
    dimension_name: str
    exact_agreement_pct: float
    within_one_pct: float
    mean_absolute_difference: float
    weighted_cohen_kappa: float
    pearson_correlation: Optional[float] = None
    spearman_correlation: Optional[float] = None
    rater1_mean: float
    rater2_mean: float


class BinaryAgreement(BaseModel):
    """Agreement statistics for binary metrics (acceptable, critical_failure)."""
    metric_name: str
    exact_agreement_pct: float
    cohen_kappa: float
    rater1_positive_count: int
    rater2_positive_count: int


class AgreementReport(BaseModel):
    """Complete agreement report between LLM Judge and Human Annotations."""
    num_evaluated_pairs: int
    rater1_name: str
    rater2_name: str
    dimension_agreements: Dict[str, DimensionAgreement] = Field(default_factory=dict)
    binary_agreements: Dict[str, BinaryAgreement] = Field(default_factory=dict)
    mean_overall_kappa: float = 0.0
    status: str = "COMPLETED"
    notes: List[str] = Field(default_factory=list)


class AutomatedResponseMetrics(BaseModel):
    """Automated deterministic response quality and safety metrics."""
    total_responses: int
    valid_structured_output_rate_pct: float
    malformed_response_rate_pct: float
    unsupported_claim_rate_pct: float
    forbidden_live_claim_rate_pct: float
    historical_date_leakage_rate_pct: float
    pii_leakage_rate_pct: float
    dangerous_false_auto_handle_count: int
    dangerous_false_auto_handle_rate_pct: float
    escalation_recall: float
    escalation_precision: float
    escalation_f1: float
    non_empty_response_rate_pct: float
    clarification_rate_pct: float
    mean_reply_word_count: float
    median_reply_word_count: float
