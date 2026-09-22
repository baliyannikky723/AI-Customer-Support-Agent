"""Deterministic Triage and Escalation Engine for AI Customer Support."""

from typing import List, Optional
from src.generation.response_schema import EvidencePack, GeneratedResponse, TriageDecision


class TriageEngine:
    """Evaluates customer state, evidence quality, and safety guardrails to decide AUTO_HANDLE vs ESCALATE."""

    # Mandatory escalation domain intents requiring secure human/carrier intervention
    HIGH_RISK_ESCALATION_INTENTS = {
        "account_access_and_security": "ACCOUNT_SECURITY: Requires verified authentication and identity recovery.",
        "order_delivered_not_received": "MISSING_PACKAGE_CLAIM: Missing delivery requires carrier investigation and claims processing.",
    }

    def __init__(
        self,
        min_intent_confidence: float = 0.60,
        min_evidence_quality: float = 0.50,
    ):
        """Initialize triage engine with conservative thresholds.

        Args:
            min_intent_confidence: Minimum intent confidence required for auto-handling.
            min_evidence_quality: Minimum evidence quality score required for auto-handling.
        """
        self.min_intent_confidence = min_intent_confidence
        self.min_evidence_quality = min_evidence_quality

    def evaluate(
        self,
        query_text: str,
        predicted_intent: str,
        confidence: float,
        evidence_pack: EvidencePack,
        generated_response: GeneratedResponse,
        guardrail_passed: bool,
        guardrail_violations: Optional[List[str]] = None,
    ) -> TriageDecision:
        """Evaluate complete interaction state and determine routing decision.

        Args:
            query_text: Raw or sanitized customer message.
            predicted_intent: Classifier predicted intent.
            confidence: Classifier confidence score.
            evidence_pack: Structured evidence pack with quality score.
            generated_response: Output from LLM response generation.
            guardrail_passed: Boolean indicating if all grounding checks passed.
            guardrail_violations: List of violation strings if any.

        Returns:
            TriageDecision with decision string, explicit reason codes, and explanation.
        """
        reason_codes: List[str] = []
        violations = guardrail_violations or []

        # Rule 1: Guardrail Failure
        if not guardrail_passed or violations:
            reason_codes.append("GUARDRAIL_VIOLATION")

        # Rule 2: Low Intent Confidence
        if confidence < self.min_intent_confidence:
            reason_codes.append("LOW_INTENT_CONFIDENCE")

        # Rule 3: Unknown or Ambiguous Intent
        if predicted_intent == "other_unknown":
            reason_codes.append("AMBIGUOUS_UNKNOWN_INTENT")

        # Rule 4: Mandatory High-Risk Safety Intents
        if predicted_intent in self.HIGH_RISK_ESCALATION_INTENTS:
            if predicted_intent == "account_access_and_security":
                reason_codes.append("ACCOUNT_SECURITY_ACTION")
            elif predicted_intent == "order_delivered_not_received":
                reason_codes.append("MISSING_PACKAGE_CLAIM")

        # Rule 5: Payment & Billing Disputes (Fraud / Double Charge)
        if predicted_intent == "payment_and_billing_issues":
            q_lower = query_text.lower()
            if any(k in q_lower for k in ["twice", "double", "fraud", "unauthorized", "stolen", "charged me"]):
                reason_codes.append("PAYMENT_DISPUTE_VERIFICATION")

        # Rule 6: Insufficient Evidence Quality
        if evidence_pack.evidence_quality_score < self.min_evidence_quality:
            reason_codes.append("INSUFFICIENT_EVIDENCE_QUALITY")

        # Rule 7: Model-Recommended Escalation or Unsupported Claims
        if generated_response.escalation_recommended:
            reason_codes.append("MODEL_ESCALATION_RECOMMENDED")
        if generated_response.unsupported_claims:
            reason_codes.append("UNSUPPORTED_CLAIMS_DETECTED")

        # Decision Determination
        if reason_codes:
            decision = "ESCALATE"
            explanation = f"Escalated to human support due to: {', '.join(reason_codes)}."
        else:
            decision = "AUTO_HANDLE"
            reason_codes.append("SAFE_GROUNDED_RESPONSE")
            explanation = "Customer inquiry safely auto-handled with policy-grounded self-service guidance."

        return TriageDecision(
            decision=decision,
            reason_codes=reason_codes,
            explanation=explanation,
            intent=predicted_intent,
            intent_confidence=round(confidence, 4),
            evidence_quality=round(evidence_pack.evidence_quality_score, 4),
            guardrail_status="PASSED" if guardrail_passed else "FAILED",
        )
