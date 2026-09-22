"""Data schemas for Evidence Packs, LLM Generated Responses, Guardrails, and Triage Decisions."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """Schema for a single historical support interaction used as evidence."""
    historical_case_id: str
    customer_issue: str
    historical_resolution: str
    similarity_score: float
    historical_intent: str
    intent_match: bool
    source_timestamp: str


class EvidencePack(BaseModel):
    """Structured pack of selected evidence for LLM prompt context."""
    query_text: str
    predicted_intent: str
    confidence: float
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    evidence_quality_score: float = 0.0
    warnings: List[str] = Field(default_factory=list)


class GeneratedResponse(BaseModel):
    """Structured contract for LLM response generation output."""
    reply: str
    grounded_claims: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    needs_clarification: bool = False
    escalation_recommended: bool = False
    escalation_reason: str = ""
    confidence: float = 1.0


class TriageDecision(BaseModel):
    """Structured output for the final routing and safety decision."""
    decision: str  # "AUTO_HANDLE" or "ESCALATE"
    reason_codes: List[str] = Field(default_factory=list)
    explanation: str
    intent: str
    intent_confidence: float
    evidence_quality: float
    guardrail_status: str  # "PASSED" or "FAILED"


class AgentResponseBundle(BaseModel):
    """Complete end-to-end trace for a single customer query interaction."""
    query_id: str
    customer_message: str
    sanitized_message: str
    predicted_intent: str
    intent_confidence: float
    evidence_pack: EvidencePack
    raw_llm_response: Optional[str] = ""
    parsed_response: GeneratedResponse
    guardrail_passed: bool
    guardrail_violations: List[str] = Field(default_factory=list)
    triage_decision: TriageDecision
    final_reply: str
    escalation_reason: Optional[str] = None
    gold_intent: Optional[str] = None
    gold_handling: Optional[str] = None
