"""Prompt Construction for Grounded Customer Support Response Generation."""

import json
from typing import Dict, Any, Optional
from src.generation.response_schema import EvidencePack


class PromptBuilder:
    """Constructs structured, safety-constrained prompts for LLM response generation."""

    SYSTEM_PROMPT = """You are the official AI Customer Support Assistant for AmazonHelp.
Your mission is to assist customers by drafting helpful, professional, and policy-grounded replies.

CRITICAL SAFETY & GROUNDING RULES:
1. Grounding in Historical Resolution Patterns:
   - Use the provided historical support cases as evidence of how AmazonHelp handles similar issues.
   - Extract general resolution principles (e.g., self-service steps in Your Orders, standard refund turnaround timelines of 5-7 business days, return window guidance).
   - Historical cases are NOT ground truth facts about the current customer. Do NOT copy historical dates (e.g. 2017 timestamps), order IDs, or customer names.

2. Forbidden Live-System Claims:
   - You DO NOT have access to live order tracking systems, payment databases, or internal carrier scans.
   - NEVER state or imply live system actions (e.g., FORBIDDEN: "I checked your order", "I can see your account", "Your refund was issued yesterday", "The carrier confirmed delivery").
   - Guide the customer on how to check their status in 'Your Orders' or 'Manage Prime Membership'.

3. Privacy & Safety:
   - Never output raw PII (emails, phone numbers, full order numbers, personal handles).
   - If an inquiry involves account security (locked account, 2FA/OTP, unauthorized access) or complex billing disputes, recommend human escalation.

4. Output Contract:
   - You MUST respond strictly in valid JSON matching the following schema:
   {
     "reply": "Clear, empathetic, and concise customer response message.",
     "grounded_claims": ["List of key policy claims supported by evidence."],
     "unsupported_claims": ["List of any unverified claims (should be empty)."],
     "needs_clarification": false,
     "escalation_recommended": false,
     "escalation_reason": "",
     "confidence": 0.95
   }
"""

    def __init__(self, system_prompt: Optional[str] = None):
        """Initialize prompt builder with optional custom system prompt."""
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT

    def build_prompt(self, evidence_pack: EvidencePack) -> Dict[str, str]:
        """Build user prompt containing customer inquiry, predicted intent, and evidence pack."""
        evidence_text_blocks = []
        for i, item in enumerate(evidence_pack.evidence_items, 1):
            block = (
                f"[Evidence Case {i}]\n"
                f"- Case ID: {item.historical_case_id}\n"
                f"- Historical Issue: {item.customer_issue}\n"
                f"- Historical Support Resolution: {item.historical_resolution}\n"
                f"- Semantic Similarity: {item.similarity_score:.4f} (Intent Match: {item.intent_match})"
            )
            evidence_text_blocks.append(block)

        evidence_section = "\n\n".join(evidence_text_blocks) if evidence_text_blocks else "No relevant historical cases found."

        user_content = f"""Customer Inquiry:
"{evidence_pack.query_text}"

Context & Intent Analysis:
- Predicted Intent: {evidence_pack.predicted_intent} (Confidence: {evidence_pack.confidence:.2f})
- Evidence Quality Score: {evidence_pack.evidence_quality_score:.2f}

Historical Resolution Evidence:
{evidence_section}

Task:
Draft a concise, empathetic customer response grounded in Amazon resolution policies.
Return ONLY valid JSON.
"""
        return {
            "system_prompt": self.system_prompt,
            "user_prompt": user_content,
        }
