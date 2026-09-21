"""PII Sanitization module for customer support messages.

Detects and masks sensitive personal identifiers including:
- Amazon Order IDs (e.g., 112-3456789-1234567, D01-1234567-1234567)
- Email addresses
- Phone numbers
- URLs with tracking/session parameters
- Anonymized numeric customer Twitter handles
"""

import re
from typing import Optional


class PIISanitizer:
    """Deterministic, regex-based PII scrubber for e-commerce customer support data."""

    # Amazon Order ID patterns (standard 3-7-7 format or digital orders)
    ORDER_ID_REGEX = re.compile(r"\b[A-Z0-9]{3}-\d{7}-\d{7}\b|\b\d{17}\b", re.IGNORECASE)

    # Standard email pattern
    EMAIL_REGEX = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        re.IGNORECASE,
    )

    # Phone number patterns (US and International formats)
    PHONE_REGEX = re.compile(
        r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        re.IGNORECASE,
    )

    # General URLs
    URL_REGEX = re.compile(
        r"https?://[^\s/$.?#].[^\s]*",
        re.IGNORECASE,
    )

    # Anonymized customer handles: e.g., @105834, @115740
    CUSTOMER_HANDLE_REGEX = re.compile(r"@\d{4,}\b")

    # Agent sign-offs: e.g., ^SN, ^MM, ^RG, -Alex
    AGENT_SIG_REGEX = re.compile(r"\s*\^[A-Z]{2,4}\b", re.IGNORECASE)

    def __init__(
        self,
        mask_order_ids: bool = True,
        mask_emails: bool = True,
        mask_phones: bool = True,
        mask_urls: bool = True,
        mask_customer_handles: bool = True,
        strip_agent_signatures: bool = True,
    ):
        self.mask_order_ids = mask_order_ids
        self.mask_emails = mask_emails
        self.mask_phones = mask_phones
        self.mask_urls = mask_urls
        self.mask_customer_handles = mask_customer_handles
        self.strip_agent_signatures = strip_agent_signatures

    def sanitize(self, text: Optional[str]) -> str:
        """Sanitize a raw text string, replacing all detected PII with tokens."""
        if not text or not isinstance(text, str):
            return ""

        sanitized = text

        # 1. Mask Email Addresses
        if self.mask_emails:
            sanitized = self.EMAIL_REGEX.sub("[EMAIL]", sanitized)

        # 2. Mask Amazon Order IDs
        if self.mask_order_ids:
            sanitized = self.ORDER_ID_REGEX.sub("[ORDER_ID]", sanitized)

        # 3. Mask Phone Numbers
        if self.mask_phones:
            sanitized = self.PHONE_REGEX.sub("[PHONE]", sanitized)

        # 4. Mask URLs
        if self.mask_urls:
            sanitized = self.URL_REGEX.sub("[URL]", sanitized)

        # 5. Mask Customer Numeric Handles
        if self.mask_customer_handles:
            sanitized = self.CUSTOMER_HANDLE_REGEX.sub("[CUSTOMER]", sanitized)

        # 6. Strip Agent Signatures
        if self.strip_agent_signatures:
            sanitized = self.AGENT_SIG_REGEX.sub("", sanitized)

        # Clean excessive whitespace
        sanitized = re.sub(r"[ \t]+", " ", sanitized).strip()

        return sanitized

    def count_pii_entities(self, text: Optional[str]) -> dict:
        """Count occurrences of each PII type in the text without modifying it."""
        if not text or not isinstance(text, str):
            return {"orders": 0, "emails": 0, "phones": 0, "urls": 0}

        return {
            "orders": len(self.ORDER_ID_REGEX.findall(text)),
            "emails": len(self.EMAIL_REGEX.findall(text)),
            "phones": len(self.PHONE_REGEX.findall(text)),
            "urls": len(self.URL_REGEX.findall(text)),
        }
