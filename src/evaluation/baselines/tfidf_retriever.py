"""Classical TF-IDF Cosine-Similarity Historical Resolution Retriever Baseline."""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.pii_sanitizer import PIISanitizer


class TFIDFRetriever:
    """Retrieves most similar historical support cases using TF-IDF cosine similarity."""

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.sanitizer = PIISanitizer()
        self.corpus_matrix: Optional[np.ndarray] = None
        self.corpus_metadata: List[Dict[str, Any]] = []
        self.is_indexed = False

    def build_index(self, corpus_df: pd.DataFrame) -> "TFIDFRetriever":
        """Index historical customer query and resolution pairs from the development corpus.
        
        Args:
            corpus_df: DataFrame containing 'conversation_id', 'customer_query', and 'amazon_response'
        """
        if len(corpus_df) == 0:
            raise ValueError("Corpus DataFrame cannot be empty")

        print(f"Building TF-IDF retrieval index across {len(corpus_df):,} historical cases...", flush=True)

        # Extract and sanitize queries and responses
        queries = []
        self.corpus_metadata = []

        for idx, row in corpus_df.reset_index(drop=True).iterrows():
            conv_id = str(row.get("conversation_id", f"case_{idx}"))
            raw_q = str(row.get("customer_query", ""))
            raw_resp = str(row.get("amazon_response", ""))

            san_q = self.sanitizer.sanitize(raw_q)
            san_resp = self.sanitizer.sanitize(raw_resp)

            queries.append(san_q)
            self.corpus_metadata.append({
                "case_id": conv_id,
                "customer_query": san_q,
                "historical_response": san_resp,
            })

        # Fit TF-IDF on historical customer queries
        self.corpus_matrix = self.vectorizer.fit_transform(queries)
        self.is_indexed = True
        print(f"TF-IDF index built successfully with vocabulary size: {len(self.vectorizer.vocabulary_):,}.")
        return self

    def retrieve(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """Retrieve top-k most similar historical cases for a single query."""
        if not self.is_indexed:
            raise RuntimeError("Retriever index must be built before retrieve()")

        san_q = self.sanitizer.sanitize(query)
        q_vec = self.vectorizer.transform([san_q])
        
        sims = cosine_similarity(q_vec, self.corpus_matrix)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            meta = self.corpus_metadata[idx]
            results.append({
                "retrieved_case_id": meta["case_id"],
                "similarity_score": round(float(sims[idx]), 4),
                "retrieved_customer_query": meta["customer_query"],
                "retrieved_historical_response": meta["historical_response"],
            })
        return results

    def batch_retrieve(self, queries: List[str], top_k: int = 1) -> List[List[Dict[str, Any]]]:
        """Perform fast vectorized batch retrieval for a list of queries."""
        if not self.is_indexed:
            raise RuntimeError("Retriever index must be built before batch_retrieve()")

        san_queries = [self.sanitizer.sanitize(q) for q in queries]
        q_matrix = self.vectorizer.transform(san_queries)
        
        # Matrix multiplication for cosine similarities
        sim_matrix = cosine_similarity(q_matrix, self.corpus_matrix)
        
        batch_results = []
        for i in range(len(queries)):
            row_sims = sim_matrix[i]
            top_indices = np.argsort(row_sims)[::-1][:top_k]
            row_results = []
            for idx in top_indices:
                meta = self.corpus_metadata[idx]
                row_results.append({
                    "retrieved_case_id": meta["case_id"],
                    "similarity_score": round(float(row_sims[idx]), 4),
                    "retrieved_customer_query": meta["customer_query"],
                    "retrieved_historical_response": meta["historical_response"],
                })
            batch_results.append(row_results)
        return batch_results
