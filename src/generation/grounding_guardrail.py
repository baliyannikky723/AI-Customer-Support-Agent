"""Deterministic Grounding, PII, and Safety Guardrails for Generated Customer Responses."""

import re
from typing import List, Tuple, Optional
from src.generation.response_schema import GeneratedResponse
from src.preprocessing.pii_sanitizer import PIISanitizer


class GroundingGuardrail:
    """Deterministic validation guardrail to detect PII leakage, hallucinations, and live-system claims."""

    # Patterns where an LLM falsely claims to perform live database/carrier lookups
    FORBIDDEN_LIVE_CLAIMS = [
        re.compile(r"\b(?:i\s+have\s+checked|i\s+checked|checking)\s+(?:your\s+order|your\s+account|the\s+status|tracking)\b", re.IGNORECASE),
        re.compile(r"\b(?:i\s+can\s+see|i\s+see)\s+(?:your\s+order|your\s+account|your\s+package)\b", re.IGNORECASE),
        re.compile(r"\b(?:your\s+refund\s+was\s+(?:issued|processed|sent)\s+on)\b", re.IGNORECASE),
        re.compile(r"\b(?:the\s+carrier\s+confirmed|fedex\s+confirmed|ups\s+confirmed|usps\s+confirmed)\b", re.IGNORECASE),
        re.compile(r"\b(?:i\s+have\s+(?:cancelled|updated|changed|refunded)\s+your)\b", re.IGNORECASE),
        re.compile(r"\b(?:your\s+package\s+is\s+currently\s+at)\b", re.IGNORECASE),
    ]

    # Stale historical timestamps from 2017 Kaggle dataset
    HISTORICAL_DATE_REGEX = re.compile(
        r"\b(?:2017|2018|2016)\b|\b(?:oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+201\d\b",
        re.IGNORECASE,
    )

    def __init__(self, sanitizer: Optional[PIISanitizer] = None):
        self.sanitizer = sanitizer or PIISanitizer()

    def validate(self, parsed_response: GeneratedResponse) -> Tuple[bool, List[str]]:
        """Validate generated response against PII, live-claim, and date leakage rules.

        Args:
            parsed_response: GeneratedResponse object from LLM generation.

        Returns:
            Tuple of (is_valid: bool, violations: List[str]).
        """
        violations = []
        reply_text = parsed_response.reply

        # 1. PII Leakage Check
        sanitized_check = self.sanitizer.sanitize(reply_text)
        if reply_text != sanitized_check and not any(tag in reply_text for tag in ["[CUSTOMER]", "[ORDER_ID]", "[EMAIL]", "[PHONE]", "[URL]"]):
            violations.append("PII_LEAKAGE_DETECTED: Raw unmasked personal identifier or order ID detected in reply.")

        # 2. Live-System Claim Check
        for pat in self.FORBIDDEN_LIVE_CLAIMS:
            if pat.search(reply_text):
                violations.append("FORBIDDEN_LIVE_CLAIM: Response falsely claims live system, order inspection, or backend action.")
                break

        # 3. Historical Date Leakage Check
        if self.HISTORICAL_DATE_REGEX.search(reply_text):
            violations.append("HISTORICAL_DATE_LEAKAGE: Response leaked stale historical 2017 timestamps from evidence.")

        # 4. Explicit Unsupported Claims in JSON
        if parsed_response.unsupported_claims:
            violations.append(f"UNSUPPORTED_CLAIMS_PRESENT: Model reported unsupported claims: {parsed_response.unsupported_claims}")

        is_valid = (len(violations) == 0)
        return is_valid, violations
