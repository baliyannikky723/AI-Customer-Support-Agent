"""Intent-Aware Retrieval Pipeline.

Coordinates Intent Classification, Dense Vector Search, and Intent-Aware Reranking
into a unified, explainable pipeline for customer support query processing.
"""

from typing import List, Optional, Dict, Any, Union
import yaml
from pathlib import Path
import pandas as pd

from src.retrieval.semantic_embedder import SemanticEmbedder
from src.retrieval.faiss_retriever import FAISSRetriever
from src.retrieval.intent_aware_reranker import IntentAwareReranker
from src.retrieval.retrieval_schema import IntentPrediction, IntentAwareBundle, RerankedResult
from src.evaluation.baselines.tfidf_intent import TFIDFIntentClassifier


class IntentAwareRetrievalPipeline:
    """Integrated Intent Classification & Intent-Aware Historical Retrieval Pipeline."""

    def __init__(
        self,
        classifier: Optional[TFIDFIntentClassifier] = None,
        retriever: Optional[FAISSRetriever] = None,
        reranker: Optional[IntentAwareReranker] = None,
        config_path: Optional[str] = "configs/default_config.yaml",
    ):
        """Initialize pipeline with components or load defaults from configuration."""
        self.config = {}
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}

        # 1. Intent Classifier
        self.classifier = classifier or TFIDFIntentClassifier()

        # 2. Vector Retriever
        if retriever is not None:
            self.retriever = retriever
        else:
            emb_cfg = self.config.get("embedding", {})
            model_name = emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            embedder = SemanticEmbedder(model_name=model_name, normalize_embeddings=True)
            self.retriever = FAISSRetriever(embedder=embedder, dimension=embedder.dimension)

        # 3. Intent Reranker
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

    def load_resources(
        self,
        index_path: str = "data/processed/faiss_index.bin",
        metadata_path: str = "data/processed/resolution_metadata.parquet",
    ) -> "IntentAwareRetrievalPipeline":
        """Load serialized FAISS vector index and metadata store."""
        self.retriever.load(index_path=index_path, metadata_path=metadata_path)
        return self

    def fit_classifier(self, X_train: List[str], y_train: List[str]) -> "IntentAwareRetrievalPipeline":
        """Fit the internal intent classifier on training texts and labels."""
        self.classifier.fit(X_train, y_train)
        return self

    def process_query(
        self,
        query_text: str,
        query_id: str = "",
        candidate_k: int = 10,
        top_k: int = 5,
        gold_intent: Optional[str] = None,
        gold_handling: Optional[str] = None,
    ) -> IntentAwareBundle:
        """Process a single incoming customer query through classification, retrieval, and reranking."""
        bundles = self.batch_process(
            query_texts=[query_text],
            query_ids=[query_id] if query_id else None,
            candidate_k=candidate_k,
            top_k=top_k,
            gold_intents=[gold_intent] if gold_intent else None,
            gold_handlings=[gold_handling] if gold_handling else None,
        )
        return bundles[0]

    def batch_process(
        self,
        query_texts: List[str],
        query_ids: Optional[List[str]] = None,
        candidate_k: int = 10,
        top_k: int = 5,
        gold_intents: Optional[List[str]] = None,
        gold_handlings: Optional[List[str]] = None,
    ) -> List[IntentAwareBundle]:
        """Batch process customer queries through the complete intent-aware retrieval pipeline."""
        if not query_texts:
            return []

        q_ids = query_ids or [f"query_{i}" for i in range(len(query_texts))]
        gold_ints = gold_intents or [None] * len(query_texts)
        gold_hands = gold_handlings or [None] * len(query_texts)

        # Step 1: Predict Intent & Confidence
        detailed_preds = self.classifier.predict_detailed(query_texts)

        # Step 2: Retrieve Candidate Pool from Vector Index
        batch_candidates = self.retriever.batch_retrieve(query_texts, top_k=candidate_k)

        bundles: List[IntentAwareBundle] = []

        for q_id, q_text, pred_data, candidates, g_int, g_hand in zip(
            q_ids, query_texts, detailed_preds, batch_candidates, gold_ints, gold_hands
        ):
            pred_intent = pred_data["predicted_intent"]
            conf = pred_data["confidence"]

            intent_pred_obj = IntentPrediction(
                predicted_intent=pred_intent,
                confidence=conf,
                probabilities=pred_data["probabilities"],
                top3=pred_data["top3"],
            )

            # Step 3: Apply Intent-Aware Reranker
            reranked_results = self.reranker.rerank(
                candidates=candidates,
                predicted_intent=pred_intent,
                confidence=conf,
                top_k=top_k,
            )

            orig_top1_id = candidates[0].conversation_id if candidates else ""
            rerank_top1_id = reranked_results[0].conversation_id if reranked_results else ""
            top1_changed = (orig_top1_id != rerank_top1_id)

            top1_sim = reranked_results[0].semantic_similarity if reranked_results else 0.0
            top1_score = reranked_results[0].final_rerank_score if reranked_results else 0.0
            top1_match = reranked_results[0].intent_match if reranked_results else False
            fallback_used = reranked_results[0].fallback_used if reranked_results else False

            bundle = IntentAwareBundle(
                query_id=q_id,
                query_text=q_text,
                intent_prediction=intent_pred_obj,
                gold_intent=g_int,
                gold_handling=g_hand,
                candidates_retrieved=len(candidates),
                top_k=len(reranked_results),
                reranking_mode=self.reranker.mode,
                fallback_used=fallback_used,
                top_1_changed=top1_changed,
                top_1_semantic_similarity=top1_sim,
                top_1_rerank_score=top1_score,
                top_1_intent_match=top1_match,
                results=reranked_results,
            )
            bundles.append(bundle)

        return bundles
