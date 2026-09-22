"""End-to-End AI Support Agent integrating Classification, RAG, Generation, Guardrails, and Triage."""

import os
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.preprocessing.pii_sanitizer import PIISanitizer
from src.evaluation.baselines.tfidf_intent import TFIDFIntentClassifier
from src.retrieval.semantic_embedder import SemanticEmbedder
from src.retrieval.faiss_retriever import FAISSRetriever
from src.retrieval.intent_aware_reranker import IntentAwareReranker
from src.generation.evidence_pack import EvidenceSelector
from src.generation.prompt_builder import PromptBuilder
from src.generation.llm_provider import LLMProvider, get_llm_provider
from src.generation.grounding_guardrail import GroundingGuardrail
from src.generation.response_generator import ResponseGenerator
from src.escalation.triage_engine import TriageEngine
from src.generation.response_schema import AgentResponseBundle, TriageDecision, GeneratedResponse


class AISupportAgent:
    """Production-grade AI Support Agent with RAG grounded generation and safety triage."""

    def __init__(
        self,
        config_path: str = "configs/default_config.yaml",
        classifier: Optional[TFIDFIntentClassifier] = None,
        retriever: Optional[FAISSRetriever] = None,
        reranker: Optional[IntentAwareReranker] = None,
        evidence_selector: Optional[EvidenceSelector] = None,
        response_generator: Optional[ResponseGenerator] = None,
        triage_engine: Optional[TriageEngine] = None,
    ):
        """Initialize agent and configure all subsystems from default_config.yaml."""
        self.config = {}
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}

        # 1. PII Sanitizer
        self.sanitizer = PIISanitizer()

        # 2. Intent Classifier
        self.classifier = classifier or TFIDFIntentClassifier()

        # 3. Vector Retriever
        if retriever is not None:
            self.retriever = retriever
        else:
            emb_cfg = self.config.get("embedding", {})
            model_name = emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            embedder = SemanticEmbedder(model_name=model_name, normalize_embeddings=True)
            self.retriever = FAISSRetriever(embedder=embedder, dimension=embedder.dimension)

        # 4. Intent Reranker
        if reranker is not None:
            self.reranker = reranker
        else:
            rerank_cfg = self.config.get("intent_reranking", {})
            bonus = rerank_cfg.get("intent_match_bonus", 0.10)
            min_conf = rerank_cfg.get("min_confidence", 0.60)
            mode = rerank_cfg.get("mode", "soft_rerank")
            self.reranker = IntentAwareReranker(
                intent_match_bonus=bonus,
                min_confidence=min_conf,
                mode=mode,
            )

        # 5. Evidence Selector
        if evidence_selector is not None:
            self.evidence_selector = evidence_selector
        else:
            gen_cfg = self.config.get("generation", {})
            max_ev = gen_cfg.get("max_evidence_items", 3)
            min_sim = gen_cfg.get("min_similarity", 0.55)
            self.evidence_selector = EvidenceSelector(
                max_evidence_items=max_ev,
                min_similarity=min_sim,
            )

        # 6. Response Generator with Guardrails
        if response_generator is not None:
            self.generator = response_generator
        else:
            llm_cfg = self.config.get("llm", {})
            provider_type = os.environ.get("LLM_PROVIDER") or llm_cfg.get("provider", "mock")
            model_name = llm_cfg.get("model", "gemini-2.5-flash")
            provider = get_llm_provider(provider_type=provider_type, model_name=model_name)
            guardrail = GroundingGuardrail(sanitizer=self.sanitizer)
            self.generator = ResponseGenerator(
                llm_provider=provider,
                guardrail=guardrail,
                max_retries=1,
            )

        # 7. Triage Engine
        if triage_engine is not None:
            self.triage_engine = triage_engine
        else:
            triage_cfg = self.config.get("triage", {})
            min_i_conf = triage_cfg.get("min_intent_confidence", 0.60)
            min_e_qual = triage_cfg.get("min_evidence_quality", 0.50)
            self.triage_engine = TriageEngine(
                min_intent_confidence=min_i_conf,
                min_evidence_quality=min_e_qual,
            )

        self._auto_load_defaults()

    def _auto_load_defaults(self):
        """Auto-load FAISS index and train classifier if default files exist and not yet loaded."""
        import pandas as pd
        index_path = Path("data/processed/faiss_index.bin")
        meta_path = Path("data/processed/resolution_metadata.parquet")
        train_path = Path("data/samples/intent_discovery_cases.parquet")

        if index_path.exists() and meta_path.exists() and self.retriever.index is None:
            try:
                self.load_resources(index_path=str(index_path), metadata_path=str(meta_path))
            except Exception as e:
                pass

        if train_path.exists() and not getattr(self.classifier, "is_fitted", False):
            try:
                df = pd.read_parquet(train_path)
                query_col = "customer_query" if "customer_query" in df.columns else ("customer_text" if "customer_text" in df.columns else None)
                if query_col:
                    X_tr = df[query_col].astype(str).tolist()
                    def _get_intent(text):
                        t = str(text).lower()
                        if any(k in t for k in ["login", "log in", "password", "locked account", "otp", "2fa", "hacked", "unauthorized"]):
                            return "account_access_and_security"
                        elif any(k in t for k in ["damaged", "broken", "crushed", "wrong item", "defective", "missing parts", "shattered", "scratched", "empty box"]):
                            return "damaged_defective_or_wrong_item"
                        elif any(k in t for k in ["says delivered", "marked delivered", "shows delivered", "not received", "missing package", "handed to resident"]):
                            return "order_delivered_not_received"
                        elif any(k in t for k in ["late", "delayed", "hasn't arrived", "still waiting", "missed date", "overdue", "running late"]):
                            return "late_delivery_complaint"
                        elif any(k in t for k in ["track", "tracking", "where is my", "dispatch", "carrier", "courier", "when will", "status"]):
                            return "delivery_status_tracking"
                        elif any(k in t for k in ["return", "pickup", "pick up", "send back", "exchange", "return label", "return policy"]):
                            return "return_and_pickup_inquiry"
                        elif any(k in t for k in ["refund", "money back", "credited", "reimburse", "bank account", "refund status"]):
                            return "refund_status_and_request"
                        elif any(k in t for k in ["cancel", "cancellation", "cancel order", "stop delivery"]):
                            return "order_cancellation_request"
                        elif any(k in t for k in ["charged twice", "double charge", "payment failed", "card declined", "debited", "gift card"]):
                            return "payment_and_billing_issues"
                        elif any(k in t for k in ["prime", "prime video", "subtitles", "alexa", "echo", "kindle"]):
                            return "prime_membership_and_digital"
                        else:
                            return "other_unknown"
                    y_tr = [_get_intent(q) for q in X_tr]
                    self.fit_classifier(X_tr, y_tr)
            except Exception as e:
                pass

    def load_resources(
        self,
        index_path: str = "data/processed/faiss_index.bin",
        metadata_path: str = "data/processed/resolution_metadata.parquet",
    ) -> "AISupportAgent":
        """Load vector index and historical resolution metadata."""
        self.retriever.load(index_path=index_path, metadata_path=metadata_path)
        return self

    def fit_classifier(self, X_train: List[str], y_train: List[str]) -> "AISupportAgent":
        """Fit the intent classifier on development training cases."""
        self.classifier.fit(X_train, y_train)
        return self

    def process_message(
        self,
        customer_message: str,
        query_id: str = "",
        gold_intent: Optional[str] = None,
        gold_handling: Optional[str] = None,
    ) -> AgentResponseBundle:
        """Execute end-to-end pipeline for a single customer query."""
        bundles = self.batch_process(
            messages=[customer_message],
            query_ids=[query_id] if query_id else None,
            gold_intents=[gold_intent] if gold_intent else None,
            gold_handlings=[gold_handling] if gold_handling else None,
        )
        return bundles[0]

    def process(self, customer_text: str, query_id: str = "") -> AgentResponseBundle:
        """Convenient alias for process_message."""
        return self.process_message(customer_message=customer_text, query_id=query_id)

    def batch_process(
        self,
        messages: List[str],
        query_ids: Optional[List[str]] = None,
        gold_intents: Optional[List[str]] = None,
        gold_handlings: Optional[List[str]] = None,
    ) -> List[AgentResponseBundle]:
        """Batch process customer queries through the complete AI Support Agent pipeline."""
        if not messages:
            return []

        q_ids = query_ids or [f"query_{i}" for i in range(len(messages))]
        gold_ints = gold_intents or [None] * len(messages)
        gold_hands = gold_handlings or [None] * len(messages)

        bundles: List[AgentResponseBundle] = []

        # Step 1: PII Sanitization of incoming customer messages
        sanitized_messages = [self.sanitizer.sanitize(m) for m in messages]

        # Step 2: Intent Classification & Calibrated Confidence
        detailed_preds = self.classifier.predict_detailed(sanitized_messages)

        # Step 3: Candidate Vector Retrieval from FAISS (k=10)
        batch_candidates = self.retriever.batch_retrieve(sanitized_messages, top_k=10)

        # Process each query
        for q_id, raw_msg, clean_msg, pred_dict, candidates, g_int, g_hand in zip(
            q_ids, messages, sanitized_messages, detailed_preds, batch_candidates, gold_ints, gold_hands
        ):
            pred_intent = pred_dict["predicted_intent"]
            confidence = pred_dict["confidence"]

            # Step 4: Intent-Aware Candidate Reranking
            reranked_results = self.reranker.rerank(
                candidates=candidates,
                predicted_intent=pred_intent,
                confidence=confidence,
                top_k=5,
            )

            # Step 5: Evidence Selection & Quality Scoring
            evidence_pack = self.evidence_selector.select_evidence(
                query_text=clean_msg,
                predicted_intent=pred_intent,
                confidence=confidence,
                candidates=reranked_results,
            )

            # Step 6: LLM Response Generation & Grounding Guardrail Validation
            parsed_response, guardrail_passed, violations, raw_llm = self.generator.generate(
                evidence_pack=evidence_pack,
            )

            # Step 7: Deterministic Triage Engine Decision
            triage_decision = self.triage_engine.evaluate(
                query_text=clean_msg,
                predicted_intent=pred_intent,
                confidence=confidence,
                evidence_pack=evidence_pack,
                generated_response=parsed_response,
                guardrail_passed=guardrail_passed,
                guardrail_violations=violations,
            )

            # Final response formatting
            final_reply = parsed_response.reply
            esc_reason = triage_decision.explanation if triage_decision.decision == "ESCALATE" else None

            bundle = AgentResponseBundle(
                query_id=q_id,
                customer_message=raw_msg,
                sanitized_message=clean_msg,
                predicted_intent=pred_intent,
                intent_confidence=confidence,
                evidence_pack=evidence_pack,
                raw_llm_response=raw_llm,
                parsed_response=parsed_response,
                guardrail_passed=guardrail_passed,
                guardrail_violations=violations,
                triage_decision=triage_decision,
                final_reply=final_reply,
                escalation_reason=esc_reason,
                gold_intent=g_int,
                gold_handling=g_hand,
            )
            bundles.append(bundle)

        return bundles
