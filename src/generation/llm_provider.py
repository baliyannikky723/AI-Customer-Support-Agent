"""LLM Provider Abstraction supporting Mock, Google Gemini, and OpenAI backends."""

import os
import re
import json
import abc
from typing import Optional, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.generation.response_schema import GeneratedResponse


class LLMProvider(abc.ABC):
    """Abstract interface for LLM response generation backends."""

    @abc.abstractmethod
    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        """Generate response text from LLM provider."""
        pass


class MockLLMProvider(LLMProvider):
    """Deterministic, offline LLM provider simulating grounded response generation."""

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        """Deterministically synthesize a grounded JSON response based on prompt context."""
        # Extract predicted intent from user prompt
        intent_match = re.search(r"Predicted Intent:\s*([a-zA-Z0-9_]+)", user_prompt)
        intent = intent_match.group(1) if intent_match else "other_unknown"

        # Extract confidence
        conf_match = re.search(r"Confidence:\s*([0-9.]+)", user_prompt)
        confidence = float(conf_match.group(1)) if conf_match else 0.80

        # Deterministic domain-grounded response templates based on AmazonHelp historical policy
        if intent == "delivery_status_tracking":
            reply = "You can track the real-time status and estimated delivery time of your package by visiting 'Your Orders' and selecting 'Track Package'. If the carrier status has not updated, please allow up to 24 hours for transit scans."
            claims = ["Live package tracking is available in 'Your Orders'", "Carrier updates may take up to 24 hours to reflect"]
            escalate = False
            esc_reason = ""
        elif intent == "late_delivery_complaint":
            reply = "We apologize for the delivery delay. Please check 'Your Orders' for the most up-to-date tracking information. If your package has not arrived within 48 hours of the guaranteed delivery date, please reach out so we can arrange a replacement or refund."
            claims = ["Delivery dates can be checked in 'Your Orders'", "Replacement or refund options apply if overdue by 48+ hours"]
            escalate = False
            esc_reason = ""
        elif intent == "order_delivered_not_received":
            reply = "We are sorry you haven't received your package. Occasionally carriers mark parcels delivered when placed in safe locations or with neighbors. If you still cannot locate it after checking around your property, please contact us for human agent escalation."
            claims = ["Packages may be left in safe locations or with neighbors", "Missing delivery requires support investigation"]
            escalate = True
            esc_reason = "Missing package marked delivered requires carrier investigation and claims filing."
        elif intent == "damaged_defective_or_wrong_item":
            reply = "We are very sorry to hear your item arrived in defective or damaged condition. You can request a free replacement or initiate a return directly through 'Your Orders' by selecting 'Return or Replace Items'."
            claims = ["Free replacement or return can be initiated in 'Your Orders'"]
            escalate = False
            esc_reason = ""
        elif intent == "return_and_pickup_inquiry":
            reply = "To schedule a return pickup or print a prepaid return label, go to 'Your Orders', choose the item, and select 'Return or Replace Items'. You will receive confirmation details and carrier pickup instructions."
            claims = ["Return pickup and prepaid labels are managed in 'Your Orders'"]
            escalate = False
            esc_reason = ""
        elif intent == "refund_status_and_request":
            reply = "Refunds are processed automatically once the returned item is received. Standard card and wallet refunds take 3-5 business days to appear, while direct bank transfers may take 5-7 business days depending on your financial institution."
            claims = ["Card refunds take 3-5 business days", "Bank transfers take 5-7 business days upon receipt"]
            escalate = False
            esc_reason = ""
        elif intent == "order_cancellation_request":
            reply = "If your order has not yet entered the shipping process, you can cancel it immediately by visiting 'Your Orders' and clicking 'Cancel Items'. If already dispatched, you can refuse the package upon delivery for a full refund."
            claims = ["Orders can be cancelled before shipping in 'Your Orders'", "Dispatched packages can be refused upon delivery"]
            escalate = False
            esc_reason = ""
        elif intent == "payment_and_billing_issues":
            reply = "For duplicate charges, temporary authorization holds are typically released by your bank within 48-72 hours. If the charge has posted or you suspect unauthorized billing, we recommend escalating to a billing specialist."
            claims = ["Authorization holds usually release within 48-72 hours", "Posted unauthorized charges require human verification"]
            escalate = True
            esc_reason = "Payment and billing disputes require account-specific verification."
        elif intent == "prime_membership_and_digital":
            reply = "You can manage, modify, or cancel your Amazon Prime membership at any time by navigating to 'Account & Lists' -> 'Your Prime Membership' and selecting 'Manage Membership'."
            claims = ["Prime settings and cancellation are accessible under 'Your Prime Membership'"]
            escalate = False
            esc_reason = ""
        elif intent == "account_access_and_security":
            reply = "For your account security, we cannot make authentication or password changes over automated messaging. Please access our official Account Recovery portal or speak with a security specialist to verify your account."
            claims = ["Account recovery requires official authentication portal"]
            escalate = True
            esc_reason = "Account access and security actions strictly require human/secure verification."
        else:
            reply = "Thank you for contacting Amazon Support. Could you please provide additional details regarding your request so we can assist you properly, or connect you with a customer service representative?"
            claims = ["General customer support inquiry"]
            escalate = True
            esc_reason = "Unrecognized or ambiguous query requires human support assistance."

        res_payload = {
            "reply": reply,
            "grounded_claims": claims,
            "unsupported_claims": [],
            "needs_clarification": (intent == "other_unknown"),
            "escalation_recommended": escalate,
            "escalation_reason": esc_reason,
            "confidence": round(confidence, 4),
        }
        return json.dumps(res_payload, indent=2)


class GeminiLLMProvider(LLMProvider):
    """Google Gemini LLM provider implementation."""

    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set.")
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        import time
        client = self._get_client()
        full_content = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        last_error = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=full_content,
                )
                return response.text or ""
            except Exception as e:
                last_error = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Gemini generation failed after 3 attempts: {last_error}")


class OpenAILLMProvider(LLMProvider):
    """OpenAI LLM provider implementation."""

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set.")
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""


def get_llm_provider(
    provider_type: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMProvider:
    """Factory function returning the configured LLM provider instance."""
    prov = (provider_type or os.environ.get("LLM_PROVIDER", "mock")).lower()

    if prov == "mock":
        return MockLLMProvider()
    elif prov in ["gemini", "google"]:
        target_model = model_name or os.environ.get("LLM_MODEL") or "gemini-3.6-flash"
        return GeminiLLMProvider(model_name=target_model, api_key=api_key)
    elif prov in ["openai", "gpt"]:
        target_model = model_name or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
        return OpenAILLMProvider(model_name=target_model, api_key=api_key)
    else:
        print(f"Warning: Unknown LLM provider '{prov}'. Falling back to MockLLMProvider.")
        return MockLLMProvider()
