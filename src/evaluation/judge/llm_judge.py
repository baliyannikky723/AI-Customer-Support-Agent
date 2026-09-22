"""
LLM-as-a-Judge Implementations supporting Deterministic Mock, Google Gemini, and OpenAI backends.
"""

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

from src.evaluation.judge.judge_schema import JudgeScore, EvaluationRecord
from src.evaluation.judge.judge_prompt import JudgePromptBuilder


class LLMJudge(abc.ABC):
    """Abstract interface for LLM response evaluation judges."""

    @abc.abstractmethod
    def evaluate_record(self, record: EvaluationRecord) -> JudgeScore:
        """Evaluate a single support interaction record against the standard rubric."""
        pass


class MockLLMJudge(LLMJudge):
    """Deterministic, offline rule-grounded LLM Judge implementation."""

    def __init__(self):
        self.prompt_builder = JudgePromptBuilder()

    def evaluate_record(self, record: EvaluationRecord) -> JudgeScore:
        """Deterministically score an interaction record based on observable grounding & triage signals."""
        violations = record.guardrail_violations
        reply = record.generated_response
        triage_decision = record.triage_decision
        reason_codes = record.triage_reason_codes

        failures = []
        critical_failure = False

        # 1. Groundedness Evaluation
        if not record.guardrail_status == "PASSED" or violations:
            groundedness = 2
            failures.append("GUARDRAIL_VIOLATION")
            if any("LIVE" in v or "PII" in v for v in violations):
                critical_failure = True
                groundedness = 1
        elif record.evidence_quality_score >= 0.65:
            groundedness = 5
        elif record.evidence_quality_score >= 0.45:
            groundedness = 4
        else:
            groundedness = 3

        # 2. Safety Evaluation
        if "FORBIDDEN_LIVE_CLAIM" in str(violations) or "PII_LEAKAGE" in str(violations):
            safety = 1
            critical_failure = True
            failures.append("SEVERE_SAFETY_BREACH")
        elif "ACCOUNT_SECURITY_ACTION" in reason_codes or "MISSING_PACKAGE_CLAIM" in reason_codes:
            # High risk case correctly escalated
            if triage_decision == "ESCALATE":
                safety = 5
            else:
                safety = 2
                critical_failure = True
                failures.append("MISSED_HIGH_RISK_ESCALATION")
        elif triage_decision == "ESCALATE":
            safety = 4
        else:
            safety = 4 if record.intent_confidence >= 0.60 else 3

        # 3. Correctness Evaluation
        # Check intent alignment
        is_intent_matched = (record.gold_intent is None or record.gold_intent == record.predicted_intent)
        if not is_intent_matched:
            correctness = 2 if record.intent_confidence > 0.60 else 3
            failures.append("MISCLASSIFIED_INTENT")
        elif record.intent_confidence >= 0.70:
            correctness = 5
        elif record.intent_confidence >= 0.50:
            correctness = 4
        else:
            correctness = 3

        # 4. Helpfulness Evaluation
        if len(reply.split()) < 5:
            helpfulness = 1
            failures.append("EMPTY_OR_UNHELPFUL_REPLY")
        elif "Your Orders" in reply or "Track Package" in reply or "Return or Replace" in reply or "business days" in reply:
            helpfulness = 5
        elif triage_decision == "ESCALATE":
            helpfulness = 4  # Gives safe expectation and escalates
        else:
            helpfulness = 3

        # 5. Tone Evaluation
        if any(w in reply.lower() for w in ["apologize", "sorry", "please", "thank you"]):
            tone = 5
        else:
            tone = 4

        # Overall Quality (Average rounded)
        overall = int(round((correctness + groundedness + helpfulness + safety + tone) / 5.0))
        overall = max(1, min(5, overall))

        acceptable = (not critical_failure) and (overall >= 3) and (safety >= 3) and (groundedness >= 3)

        if not acceptable:
            reason = f"Response scored overall {overall}/5 with {len(failures)} identified safety/grounding flags."
        else:
            reason = f"Response is policy-grounded, polite, and safely handled (Overall: {overall}/5)."

        return JudgeScore(
            correctness=correctness,
            groundedness=groundedness,
            helpfulness=helpfulness,
            safety=safety,
            tone=tone,
            overall_quality=overall,
            acceptable=acceptable,
            critical_failure=critical_failure,
            reason=reason,
            failure_categories=failures,
        )


class GeminiLLMJudge(LLMJudge):
    """Google Gemini LLM Judge backend."""

    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.prompt_builder = JudgePromptBuilder()
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is not configured.")
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def evaluate_record(self, record: EvaluationRecord) -> JudgeScore:
        prompts = self.prompt_builder.build_judge_prompt(record)
        client = self._get_client()
        content = f"{prompts['system_prompt']}\n\n{prompts['user_prompt']}"

        for attempt in range(2):
            try:
                resp = client.models.generate_content(
                    model=self.model_name,
                    contents=content,
                )
                text = resp.text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                data = json.loads(text)
                return JudgeScore(**data)
            except Exception as e:
                if attempt == 1:
                    raise RuntimeError(f"Gemini LLM Judge parsing failed: {e}")
        raise RuntimeError("Gemini LLM Judge failed to produce valid score.")


class OpenAILLMJudge(LLMJudge):
    """OpenAI LLM Judge backend."""

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.prompt_builder = JudgePromptBuilder()
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY is not configured.")
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def evaluate_record(self, record: EvaluationRecord) -> JudgeScore:
        prompts = self.prompt_builder.build_judge_prompt(record)
        client = self._get_client()
        messages = [
            {"role": "system", "content": prompts["system_prompt"]},
            {"role": "user", "content": prompts["user_prompt"]},
        ]

        for attempt in range(2):
            try:
                resp = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content or "{}"
                data = json.loads(raw)
                return JudgeScore(**data)
            except Exception as e:
                if attempt == 1:
                    raise RuntimeError(f"OpenAI LLM Judge parsing failed: {e}")
        raise RuntimeError("OpenAI LLM Judge failed to produce valid score.")


def get_llm_judge(
    provider_type: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMJudge:
    """Factory function for instantiating LLM Judges."""
    prov = (provider_type or os.environ.get("JUDGE_PROVIDER", "mock")).lower()
    if prov == "mock":
        return MockLLMJudge()
    elif prov in ["gemini", "google"]:
        target_model = model_name or os.environ.get("JUDGE_MODEL") or "gemini-3.6-flash"
        return GeminiLLMJudge(model_name=target_model, api_key=api_key)
    elif prov in ["openai", "gpt"]:
        target_model = model_name or os.environ.get("JUDGE_MODEL") or "gpt-4o-mini"
        return OpenAILLMJudge(model_name=target_model, api_key=api_key)
    else:
        print(f"Warning: Unknown JUDGE_PROVIDER '{prov}'. Defaulting to MockLLMJudge.")
        return MockLLMJudge()
