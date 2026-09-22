"""Response Generator orchestrating prompt construction, LLM generation, and guardrail validation."""

import json
import re
from typing import Optional, Tuple, List

from src.generation.response_schema import EvidencePack, GeneratedResponse
from src.generation.prompt_builder import PromptBuilder
from src.generation.llm_provider import LLMProvider, get_llm_provider
from src.generation.grounding_guardrail import GroundingGuardrail


class ResponseGenerator:
    """Coordinates prompt formatting, LLM inference, JSON parsing, and grounding validation."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        guardrail: Optional[GroundingGuardrail] = None,
        max_retries: int = 1,
    ):
        """Initialize response generator with provider and guardrails.

        Args:
            llm_provider: LLMProvider instance (defaults to configured/mock provider).
            prompt_builder: PromptBuilder instance.
            guardrail: GroundingGuardrail instance.
            max_retries: Number of regeneration attempts on guardrail failure.
        """
        self.provider = llm_provider or get_llm_provider()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.guardrail = guardrail or GroundingGuardrail()
        self.max_retries = max_retries

    def generate(
        self,
        evidence_pack: EvidencePack,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> Tuple[GeneratedResponse, bool, List[str], str]:
        """Generate, parse, and validate customer response.

        Args:
            evidence_pack: EvidencePack containing customer query and selected historical cases.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.

        Returns:
            Tuple of (GeneratedResponse, guardrail_passed: bool, violations: List[str], raw_text: str).
        """
        prompt_dict = self.prompt_builder.build_prompt(evidence_pack)
        user_prompt = prompt_dict["user_prompt"]
        system_prompt = prompt_dict["system_prompt"]

        raw_output = self.provider.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        parsed_response = self._parse_json_response(raw_output, evidence_pack)
        is_valid, violations = self.guardrail.validate(parsed_response)

        # Optional single retry if validation fails and retries allowed
        if not is_valid and self.max_retries > 0:
            strict_prompt = (
                f"{user_prompt}\n\n"
                f"PREVIOUS ATTEMPT FAILED GUARDRAIL CHECK:\n"
                f"{'; '.join(violations)}\n"
                f"Please fix all issues: Remove any forbidden live claims, stale dates, or PII. Return valid JSON only."
            )
            raw_output = self.provider.generate(
                user_prompt=strict_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            parsed_response = self._parse_json_response(raw_output, evidence_pack)
            is_valid, violations = self.guardrail.validate(parsed_response)

        return parsed_response, is_valid, violations, raw_output

    def _parse_json_response(self, raw_text: str, evidence_pack: EvidencePack) -> GeneratedResponse:
        """Parse raw LLM output string into GeneratedResponse Pydantic object."""
        try:
            # Clean markdown codeblocks if present
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
            data = json.loads(cleaned)
            return GeneratedResponse(**data)
        except Exception as e:
            # Safe fallback if model output failed JSON parsing
            return GeneratedResponse(
                reply="Thank you for reaching out to Amazon Customer Service. Please review your order status under 'Your Orders' or contact a support specialist for assistance.",
                grounded_claims=["Self-service orders available under 'Your Orders'"],
                unsupported_claims=[f"JSON_PARSE_ERROR: {str(e)}"],
                needs_clarification=True,
                escalation_recommended=True,
                escalation_reason=f"LLM output formatting error: {str(e)}",
                confidence=0.50,
            )
