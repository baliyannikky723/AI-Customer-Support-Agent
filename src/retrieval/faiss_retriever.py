"""Dense Vector Retrieval using FAISS and Sentence-Transformers.

Implements exact cosine similarity nearest-neighbor search with normalized vectors,
rich historical metadata storage, PII sanitization guardrails, and serialization.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import faiss
import numpy as np
import pandas as pd

from src.retrieval.semantic_embedder import SemanticEmbedder
from src.retrieval.retrieval_schema import RetrievalResult
from src.preprocessing.pii_sanitizer import PIISanitizer


class FAISSRetriever:
    """FAISS-based Dense Semantic Retriever for historical customer support cases."""

    def __init__(
        self,
        embedder: Optional[SemanticEmbedder] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        pii_sanitizer: Optional[PIISanitizer] = None,
    ):
        """Initialize FAISS retriever.

        Args:
            embedder: Pre-instantiated SemanticEmbedder (or None to create one).
            model_name: Model name if embedder is not passed.
            dimension: Embedding vector dimension (384 for MiniLM).
            pii_sanitizer: PIISanitizer instance for scrubbing outputs.
        """
        self.embedder = embedder or SemanticEmbedder(model_name=model_name)
        self.dimension = dimension or self.embedder.dimension
        self.sanitizer = pii_sanitizer or PIISanitizer()
        
        # FAISS IndexFlatIP implements exact Inner Product search.
        # When vectors are L2-normalized, Inner Product == Cosine Similarity.
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []

    @property
    def is_indexed(self) -> bool:
        """Return True if index is built and non-empty."""
        return self.index is not None and self.index.ntotal > 0

    @property
    def size(self) -> int:
        """Return number of indexed items."""
        return self.index.ntotal if self.index is not None else 0

    def build_index(
        self,
        corpus_df: pd.DataFrame,
        query_col: str = "customer_query",
        reply_col: str = "amazon_response",
        id_col: str = "conversation_id",
        timestamp_col: str = "start_time",
        intent_col: Optional[str] = None,
        deduplicate: bool = True,
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> int:
        """Build FAISS IndexFlatIP index over a corpus DataFrame.

        Args:
            corpus_df: DataFrame containing historical support conversations.
            query_col: Column name containing customer inquiry text.
            reply_col: Column name containing support resolution text.
            id_col: Column name for conversation ID.
            timestamp_col: Column name for timestamp.
            intent_col: Optional column name for intent (for analysis only).
            deduplicate: If True, deduplicates exact (query, reply) matches.
            batch_size: Batch size for embedding model.
            show_progress: If True, displays progress bar.

        Returns:
            Number of indexed records.
        """
        df = corpus_df.copy()

        # Deduplicate exact duplicate query-reply pairs if requested
        if deduplicate:
            initial_len = len(df)
            df = df.drop_duplicates(subset=[query_col, reply_col]).reset_index(drop=True)
            print(f"Corpus deduplication: {initial_len:,} -> {len(df):,} unique records.")

        texts_to_embed = df[query_col].astype(str).tolist()

        # Generate L2-normalized dense embeddings
        embeddings = self.embedder.encode(
            texts_to_embed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )

        # Create FAISS IndexFlatIP
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        # Build associated metadata store with PII sanitization
        self.metadata = []
        for idx, row in df.iterrows():
            raw_reply = str(row.get(reply_col, ""))
            sanitized_reply = self.sanitizer.sanitize(raw_reply)

            meta_item = {
                "index_id": int(idx),
                "conversation_id": str(row.get(id_col, "")),
                "customer_message": str(row.get(query_col, "")),
                "historical_reply": sanitized_reply,
                "timestamp": str(row.get(timestamp_col, "")),
                "intent": str(row.get(intent_col, "other_unknown")) if intent_col and intent_col in row else "other_unknown",
                "num_turns": int(row.get("num_turns", 2)) if "num_turns" in row else 2,
            }
            self.metadata.append(meta_item)

        return self.index.ntotal

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Retrieve top-k historical support cases for a single query.

        Args:
            query: Customer query text.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of RetrievalResult objects sorted by similarity score descending.
        """
        results = self.batch_retrieve([query], top_k=top_k)
        return results[0] if results else []

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
    ) -> List[List[RetrievalResult]]:
        """Retrieve top-k historical support cases for a batch of queries.

        Args:
            queries: List of customer query strings.
            top_k: Number of nearest neighbors to return per query.

        Returns:
            List of lists of RetrievalResult objects.
        """
        if not self.is_indexed:
            raise RuntimeError("Retriever index has not been built or loaded.")

        if not queries:
            return []

        # Encode queries (L2 normalized)
        query_embeddings = self.embedder.encode(
            queries,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Execute FAISS search (distances are inner products == cosine similarity)
        k_search = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embeddings, k_search)

        all_results: List[List[RetrievalResult]] = []

        for q_idx in range(len(queries)):
            query_res: List[RetrievalResult] = []
            for rank_idx in range(k_search):
                match_idx = int(indices[q_idx][rank_idx])
                sim_score = float(scores[q_idx][rank_idx])

                if match_idx < 0 or match_idx >= len(self.metadata):
                    continue

                meta = self.metadata[match_idx]
                result_obj = RetrievalResult(
                    conversation_id=meta["conversation_id"],
                    customer_message=meta["customer_message"],
                    historical_reply=meta["historical_reply"],
                    similarity_score=round(sim_score, 4),
                    timestamp=meta["timestamp"],
                    intent=meta.get("intent", "other_unknown"),
                    rank=rank_idx + 1,
                )
                query_res.append(result_obj)
            all_results.append(query_res)

        return all_results

    def save(self, index_path: Union[str, Path], metadata_path: Union[str, Path]) -> None:
        """Serialize FAISS binary index and metadata parquet to disk."""
        if not self.is_indexed:
            raise RuntimeError("Cannot save uninitialized FAISS index.")

        p_idx = Path(index_path)
        p_idx.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p_idx))

        p_meta = Path(metadata_path)
        p_meta.parent.mkdir(parents=True, exist_ok=True)
        df_meta = pd.DataFrame(self.metadata)
        df_meta.to_parquet(p_meta, index=False)

    def load(self, index_path: Union[str, Path], metadata_path: Union[str, Path]) -> None:
        """Load serialized FAISS index and metadata from disk."""
        p_idx = Path(index_path)
        p_meta = Path(metadata_path)

        if not p_idx.exists():
            raise FileNotFoundError(f"FAISS index file not found: {p_idx}")
        if not p_meta.exists():
            raise FileNotFoundError(f"Metadata file not found: {p_meta}")

        self.index = faiss.read_index(str(p_idx))
        df_meta = pd.read_parquet(p_meta)
        self.metadata = df_meta.to_dict(orient="records")
