"""Retrieval and Intent-Aware Reranking Data Schemas.

Defines Pydantic data structures for historical indexed cases,
retrieval query requests, classifier confidence outputs, and reranked candidate bundles.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class IndexedCase(BaseModel):
    """Schema for a historical customer support interaction in the vector index."""
    conversation_id: str
    customer_message: str
    historical_reply: str
    timestamp: str
    context: Optional[str] = ""
    intent: Optional[str] = "other_unknown"
    is_canned: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Schema for a single retrieved historical case match."""
    conversation_id: str
    customer_message: str
    historical_reply: str
    similarity_score: float
    timestamp: str
    intent: Optional[str] = "other_unknown"
    context: Optional[str] = ""
    is_canned: Optional[bool] = False
    rank: int = 1


class QueryRetrievalBundle(BaseModel):
    """Schema for top-k retrieval results for a specific query."""
    query_id: str
    query_text: str
    gold_intent: Optional[str] = None
    results: List[RetrievalResult] = Field(default_factory=list)
    top_1_similarity: float = 0.0
    mean_top_k_similarity: float = 0.0
    same_intent_top_1: Optional[bool] = None
    same_intent_in_top_k: Optional[bool] = None
    unique_response_templates: int = 1


class IntentPrediction(BaseModel):
    """Schema for classifier prediction and confidence distribution."""
    predicted_intent: str
    confidence: float
    probabilities: Dict[str, float] = Field(default_factory=dict)
    top3: List[Dict[str, Any]] = Field(default_factory=list)


class RerankedResult(BaseModel):
    """Schema for an intent-aware reranked historical case match."""
    conversation_id: str
    customer_message: str
    historical_reply: str
    semantic_similarity: float
    predicted_intent: str
    historical_intent: str
    intent_match: bool
    intent_bonus: float
    final_rerank_score: float
    timestamp: str
    original_rank: int
    reranked_rank: int
    fallback_used: bool = False
    is_canned: Optional[bool] = False


class IntentAwareBundle(BaseModel):
    """Schema for the full intent-aware retrieval output bundle."""
    query_id: str
    query_text: str
    intent_prediction: IntentPrediction
    gold_intent: Optional[str] = None
    gold_handling: Optional[str] = None
    candidates_retrieved: int = 10
    top_k: int = 5
    reranking_mode: str = "soft_rerank"
    fallback_used: bool = False
    top_1_changed: bool = False
    top_1_semantic_similarity: float = 0.0
    top_1_rerank_score: float = 0.0
    top_1_intent_match: bool = False
    results: List[RerankedResult] = Field(default_factory=list)
