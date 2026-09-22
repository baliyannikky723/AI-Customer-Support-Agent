"""Evidence Selection and Quality Scoring for Grounded Response Generation."""

from typing import List, Optional
import numpy as np

from src.retrieval.retrieval_schema import RerankedResult
from src.generation.response_schema import EvidenceItem, EvidencePack


class EvidenceSelector:
    """Filters, deduplicates, and scores historical support cases for LLM generation context."""

    def __init__(
        self,
        max_evidence_items: int = 3,
        min_similarity: float = 0.55,
        deduplicate_templates: bool = True,
    ):
        """Initialize evidence selector.

        Args:
            max_evidence_items: Maximum number of historical cases to pass in prompt.
            min_similarity: Minimum raw cosine similarity threshold for inclusion.
            deduplicate_templates: If True, filters out duplicate resolution strings.
        """
        self.max_evidence_items = max_evidence_items
        self.min_similarity = min_similarity
        self.deduplicate_templates = deduplicate_templates
        
        # Generic boilerplate deflection markers
        self.generic_markers = [
            "please reach out to us here",
            "please contact us here",
            "we'd like to look into this",
            "send us a dm",
            "direct message",
        ]

    def select_evidence(
        self,
        query_text: str,
        predicted_intent: str,
        confidence: float,
        candidates: List[RerankedResult],
    ) -> EvidencePack:
        """Select top informative, diverse, and intent-compatible historical cases.

        Args:
            query_text: Customer query string.
            predicted_intent: Classifier predicted intent.
            confidence: Classifier confidence score.
            candidates: List of reranked retrieval results.

        Returns:
            EvidencePack containing structured items, quality score, and warnings.
        """
        selected_items: List[EvidenceItem] = []
        seen_templates = set()
        warnings: List[str] = []

        # 1. Filter candidates by similarity threshold and deduplicate
        for c in candidates:
            if c.semantic_similarity < self.min_similarity:
                continue

            cleaned_reply = c.historical_reply.strip().lower()
            if self.deduplicate_templates and cleaned_reply in seen_templates:
                continue
            seen_templates.add(cleaned_reply)

            item = EvidenceItem(
                historical_case_id=c.conversation_id,
                customer_issue=c.customer_message,
                historical_resolution=c.historical_reply,
                similarity_score=round(c.semantic_similarity, 4),
                historical_intent=c.historical_intent,
                intent_match=c.intent_match,
                source_timestamp=c.timestamp,
            )
            selected_items.append(item)

            if len(selected_items) >= self.max_evidence_items:
                break

        # If strict filtering produced 0 items, fallback to top-1 raw candidate if available
        if not selected_items and candidates:
            top_cand = candidates[0]
            warnings.append(f"All candidates below min_similarity ({self.min_similarity}). Used top fallback candidate.")
            selected_items.append(
                EvidenceItem(
                    historical_case_id=top_cand.conversation_id,
                    customer_issue=top_cand.customer_message,
                    historical_resolution=top_cand.historical_reply,
                    similarity_score=round(top_cand.semantic_similarity, 4),
                    historical_intent=top_cand.historical_intent,
                    intent_match=top_cand.intent_match,
                    source_timestamp=top_cand.timestamp,
                )
            )

        # 2. Compute Deterministic Evidence Quality Score
        quality_score, quality_warnings = self.compute_evidence_quality_score(
            items=selected_items,
            predicted_intent=predicted_intent,
        )
        warnings.extend(quality_warnings)

        return EvidencePack(
            query_text=query_text,
            predicted_intent=predicted_intent,
            confidence=round(confidence, 4),
            evidence_items=selected_items,
            evidence_quality_score=quality_score,
            warnings=warnings,
        )

    def compute_evidence_quality_score(
        self,
        items: List[EvidenceItem],
        predicted_intent: str,
    ) -> tuple[float, List[str]]:
        """Compute transparent deterministic evidence quality score in range [0.0, 1.0].

        Formula:
            Score = (0.40 * Top1_Sim) + (0.30 * Mean_Sim) + (0.20 * Intent_Match_Ratio) + (0.10 * Item_Count_Ratio)
                    - Penalties (e.g. all generic boilerplate)
        """
        warnings = []
        if not items:
            return 0.0, ["No historical evidence items available."]

        top1_sim = items[0].similarity_score
        mean_sim = float(np.mean([item.similarity_score for item in items]))
        match_count = sum(1 for item in items if item.historical_intent == predicted_intent)
        match_ratio = match_count / len(items)
        count_ratio = min(len(items) / self.max_evidence_items, 1.0)

        raw_score = (0.40 * top1_sim) + (0.30 * mean_sim) + (0.20 * match_ratio) + (0.10 * count_ratio)

        # Check for generic boilerplate penalty
        generic_count = sum(1 for item in items if any(m in item.historical_resolution.lower() for m in self.generic_markers))
        if generic_count == len(items):
            raw_score -= 0.20
            warnings.append("All selected evidence items are generic deflection boilerplate.")

        if match_count == 0:
            warnings.append(f"Zero evidence items match predicted intent '{predicted_intent}'.")

        final_score = float(np.clip(raw_score, 0.0, 1.0))
        return round(final_score, 4), warnings
