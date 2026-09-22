"""Intent-Aware Candidate Reranker for Historical Support Retrieval.

Combines dense vector semantic similarity with predicted intent compatibility,
providing configurable soft-reranking bonuses, low-confidence fallback guardrails,
and explainable scoring breakdowns.
"""

from typing import List, Optional, Union
from src.retrieval.retrieval_schema import RetrievalResult, RerankedResult


class IntentAwareReranker:
    """Reranks semantic nearest neighbors using classifier intent compatibility."""

    def __init__(
        self,
        intent_match_bonus: float = 0.10,
        min_confidence: float = 0.60,
        mode: str = "soft_rerank",
    ):
        """Initialize intent-aware reranker.

        Args:
            intent_match_bonus: Additive score bonus when historical intent matches predicted intent.
            min_confidence: Minimum classifier confidence required to apply intent reranking.
            mode: Reranking strategy - 'soft_rerank' (default) or 'hard_intent_filter'.
        """
        if mode not in ["soft_rerank", "hard_intent_filter"]:
            raise ValueError(f"Unsupported reranking mode '{mode}'. Choose 'soft_rerank' or 'hard_intent_filter'.")

        self.intent_match_bonus = intent_match_bonus
        self.min_confidence = min_confidence
        self.mode = mode

    def rerank(
        self,
        candidates: List[RetrievalResult],
        predicted_intent: str,
        confidence: float,
        top_k: int = 5,
    ) -> List[RerankedResult]:
        """Rerank retrieval candidates based on predicted intent alignment.

        Args:
            candidates: List of RetrievalResult objects from vector retrieval.
            predicted_intent: The intent predicted by the intent classifier.
            confidence: The prediction probability / confidence of the classifier.
            top_k: Number of top reranked candidates to return.

        Returns:
            List of RerankedResult objects sorted descending by final_rerank_score.
        """
        if not candidates:
            return []

        # Check for Low-Confidence Fallback Guardrail
        use_fallback = confidence < self.min_confidence

        reranked_items: List[RerankedResult] = []

        if use_fallback:
            # Low confidence: preserve raw semantic ranking without intent bonus
            for orig_rank, c in enumerate(candidates, 1):
                hist_intent = c.intent or "other_unknown"
                is_match = (hist_intent == predicted_intent)
                
                item = RerankedResult(
                    conversation_id=c.conversation_id,
                    customer_message=c.customer_message,
                    historical_reply=c.historical_reply,
                    semantic_similarity=c.similarity_score,
                    predicted_intent=predicted_intent,
                    historical_intent=hist_intent,
                    intent_match=is_match,
                    intent_bonus=0.0,
                    final_rerank_score=c.similarity_score,
                    timestamp=c.timestamp,
                    original_rank=orig_rank,
                    reranked_rank=orig_rank,
                    fallback_used=True,
                    is_canned=c.is_canned,
                )
                reranked_items.append(item)
            return reranked_items[:top_k]

        if self.mode == "hard_intent_filter":
            # Hard filter: select only candidates with matching intent
            matching_candidates = [c for c in candidates if (c.intent or "other_unknown") == predicted_intent]
            active_pool = matching_candidates if matching_candidates else candidates
            hard_fallback = len(matching_candidates) == 0

            for orig_rank, c in enumerate(active_pool, 1):
                hist_intent = c.intent or "other_unknown"
                is_match = (hist_intent == predicted_intent)
                item = RerankedResult(
                    conversation_id=c.conversation_id,
                    customer_message=c.customer_message,
                    historical_reply=c.historical_reply,
                    semantic_similarity=c.similarity_score,
                    predicted_intent=predicted_intent,
                    historical_intent=hist_intent,
                    intent_match=is_match,
                    intent_bonus=self.intent_match_bonus if is_match else 0.0,
                    final_rerank_score=c.similarity_score + (self.intent_match_bonus if is_match else 0.0),
                    timestamp=c.timestamp,
                    original_rank=c.rank,
                    reranked_rank=orig_rank,
                    fallback_used=hard_fallback,
                    is_canned=c.is_canned,
                )
                reranked_items.append(item)

            reranked_items.sort(key=lambda x: x.final_rerank_score, reverse=True)
            for new_rank, item in enumerate(reranked_items, 1):
                item.reranked_rank = new_rank
            return reranked_items[:top_k]

        # Default Mode: Soft Reranking with additive bonus
        for orig_rank, c in enumerate(candidates, 1):
            hist_intent = c.intent or "other_unknown"
            is_match = (hist_intent == predicted_intent)
            bonus = self.intent_match_bonus if is_match else 0.0
            score = round(c.similarity_score + bonus, 4)

            item = RerankedResult(
                conversation_id=c.conversation_id,
                customer_message=c.customer_message,
                historical_reply=c.historical_reply,
                semantic_similarity=c.similarity_score,
                predicted_intent=predicted_intent,
                historical_intent=hist_intent,
                intent_match=is_match,
                intent_bonus=bonus,
                final_rerank_score=score,
                timestamp=c.timestamp,
                original_rank=orig_rank,
                reranked_rank=orig_rank,
                fallback_used=False,
                is_canned=c.is_canned,
            )
            reranked_items.append(item)

        # Sort descending by final_rerank_score, tie-breaking by raw similarity
        reranked_items.sort(key=lambda x: (x.final_rerank_score, x.semantic_similarity), reverse=True)

        for new_rank, item in enumerate(reranked_items, 1):
            item.reranked_rank = new_rank

        return reranked_items[:top_k]
