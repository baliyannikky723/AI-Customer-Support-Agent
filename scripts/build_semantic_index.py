"""Build Semantic FAISS Retrieval Index over Historical AmazonHelp Support Cases.

Strictly indexes the isolated development partition (pre-cutoff 85%), ensures zero leakage
with the golden evaluation set, deduplicates exact query-reply pairs, and serializes the
dense vector index (IndexFlatIP) and sanitized metadata store to disk.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any
import yaml
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.semantic_embedder import SemanticEmbedder
from src.retrieval.faiss_retriever import FAISSRetriever
from scripts.validate_golden_isolation import validate_golden_isolation


def build_semantic_index(
    corpus_path: str = "data/samples/intent_discovery_cases.parquet",
    output_index_path: str = "data/processed/faiss_index.bin",
    output_metadata_path: str = "data/processed/resolution_metadata.parquet",
    golden_inputs_path: str = "data/golden/golden_inputs.jsonl",
    golden_labels_path: str = "data/golden/golden_labels.jsonl",
    config_path: str = "configs/default_config.yaml",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    deduplicate: bool = True,
) -> Dict[str, Any]:
    """Build and serialize the FAISS semantic retrieval index."""
    start_total_time = time.time()

    # Load configuration if available
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    emb_cfg = cfg.get("embedding", {})
    ret_cfg = cfg.get("retrieval", {})

    model_name = emb_cfg.get("model_name", model_name)
    batch_size = emb_cfg.get("batch_size", batch_size)
    output_index_path = ret_cfg.get("faiss_index_path", output_index_path)
    output_metadata_path = ret_cfg.get("metadata_path", output_metadata_path)

    print("=" * 60)
    print("PHASE 6: BUILDING SEMANTIC HISTORICAL RETRIEVAL INDEX")
    print("=" * 60)
    print(f"Embedding Model       : {model_name}")
    print(f"Embedding Dimension   : 384")
    print(f"FAISS Index Type      : IndexFlatIP (Cosine Similarity on L2-Norm)")
    print(f"Corpus Path           : {corpus_path}")
    print(f"Output Index Path     : {output_index_path}")
    print(f"Output Metadata Path  : {output_metadata_path}")
    print("=" * 60, flush=True)

    # Step 1: Leakage Audit Guardrails
    print("\n[Step 1/5] Executing Golden Set Isolation Audit...", flush=True)
    isolation_res = validate_golden_isolation(
        golden_inputs_path=golden_inputs_path,
        golden_labels_path=golden_labels_path,
        intent_cases_path=corpus_path,
    )
    if isolation_res["status"] != "PASSED":
        raise RuntimeError("CRITICAL LEAKAGE DETECTED: Corpus overlaps with Golden Set! Aborting indexing.")
    print("Isolation audit verified: 0 golden conversations in development corpus.\n", flush=True)

    # Step 2: Load Development Corpus
    print(f"[Step 2/5] Loading historical corpus from {corpus_path}...", flush=True)
    df_corpus = pd.read_parquet(corpus_path)
    raw_corpus_size = len(df_corpus)
    print(f"Loaded {raw_corpus_size:,} historical support interactions.", flush=True)

    # Step 3: Deduplication and Canned Template Analysis
    print("\n[Step 3/5] Analyzing duplicates and canned templates...", flush=True)
    initial_count = len(df_corpus)
    if deduplicate:
        df_corpus = df_corpus.drop_duplicates(subset=["customer_query", "amazon_response"]).reset_index(drop=True)
        dedup_count = len(df_corpus)
        exact_duplicates_removed = initial_count - dedup_count
        print(f"Removed {exact_duplicates_removed:,} exact duplicate (query, response) pairs. Active index pool: {dedup_count:,}")
    else:
        exact_duplicates_removed = 0
        dedup_count = len(df_corpus)

    # Measure top response frequency
    top_responses = df_corpus["amazon_response"].value_counts()
    top_canned_pct = (top_responses.iloc[0] / len(df_corpus)) * 100 if len(top_responses) > 0 else 0.0
    unique_responses = len(top_responses)
    print(f"Unique resolution responses: {unique_responses:,} / {dedup_count:,} ({unique_responses/dedup_count*100:.1f}%)")
    print(f"Most frequent response occurs {top_responses.iloc[0]} times ({top_canned_pct:.2f}% of corpus).", flush=True)

    # Inferred historical intent annotation (strictly rule-based on dev data, no golden leakage)
    def infer_intent(text):
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

    df_corpus["intent"] = [infer_intent(q) for q in df_corpus["customer_query"]]

    # Step 4: Semantic Embedding & Index Creation
    print(f"\n[Step 4/5] Encoding {dedup_count:,} queries using {model_name}...", flush=True)
    embedder = SemanticEmbedder(model_name=model_name, batch_size=batch_size, normalize_embeddings=True)
    retriever = FAISSRetriever(embedder=embedder, dimension=embedder.dimension)

    start_embed_time = time.time()
    retriever.build_index(
        corpus_df=df_corpus,
        query_col="customer_query",
        reply_col="amazon_response",
        id_col="conversation_id",
        timestamp_col="start_time",
        intent_col="intent",
        deduplicate=False,  # Already deduplicated above
        batch_size=batch_size,
        show_progress=True,
    )
    embed_time = time.time() - start_embed_time
    print(f"Embedded and indexed {retriever.size:,} records in {embed_time:.2f}s ({retriever.size / max(embed_time, 0.001):.1f} cases/sec).", flush=True)

    # Step 5: Serialize Index and Metadata
    print(f"\n[Step 5/5] Serializing FAISS index and metadata to disk...", flush=True)
    start_save_time = time.time()
    retriever.save(index_path=output_index_path, metadata_path=output_metadata_path)
    save_time = time.time() - start_save_time

    index_size_bytes = os.path.getsize(output_index_path)
    metadata_size_bytes = os.path.getsize(output_metadata_path)
    total_time = time.time() - start_total_time

    summary = {
        "status": "SUCCESS",
        "model_name": model_name,
        "embedding_dimension": embedder.dimension,
        "index_type": "IndexFlatIP",
        "raw_corpus_size": raw_corpus_size,
        "indexed_records": retriever.size,
        "exact_duplicates_removed": exact_duplicates_removed,
        "unique_resolutions": unique_responses,
        "most_frequent_response_pct": round(top_canned_pct, 2),
        "embedding_build_time_seconds": round(embed_time, 2),
        "serialization_time_seconds": round(save_time, 2),
        "total_elapsed_seconds": round(total_time, 2),
        "index_file_size_mb": round(index_size_bytes / (1024 * 1024), 2),
        "metadata_file_size_mb": round(metadata_size_bytes / (1024 * 1024), 2),
        "output_index_path": str(output_index_path),
        "output_metadata_path": str(output_metadata_path),
    }

    print("\n" + "=" * 60)
    print("SEMANTIC INDEX BUILD COMPLETED SUCCESSFULLY")
    print(f"Total Indexed Cases   : {retriever.size:,}")
    print(f"FAISS Index Size      : {summary['index_file_size_mb']} MB")
    print(f"Metadata Store Size   : {summary['metadata_file_size_mb']} MB")
    print(f"Embedding Build Time  : {summary['embedding_build_time_seconds']}s")
    print(f"Total Build Time      : {summary['total_elapsed_seconds']}s")
    print("=" * 60 + "\n", flush=True)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS semantic retrieval index.")
    parser.add_argument("--corpus-path", type=str, default="data/samples/intent_discovery_cases.parquet")
    parser.add_argument("--output-index", type=str, default="data/processed/faiss_index.bin")
    parser.add_argument("--output-metadata", type=str, default="data/processed/resolution_metadata.parquet")
    parser.add_argument("--golden-inputs", type=str, default="data/golden/golden_inputs.jsonl")
    parser.add_argument("--golden-labels", type=str, default="data/golden/golden_labels.jsonl")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()
    build_semantic_index(
        corpus_path=args.corpus_path,
        output_index_path=args.output_index,
        output_metadata_path=args.output_metadata,
        golden_inputs_path=args.golden_inputs,
        golden_labels_path=args.golden_labels,
        config_path=args.config,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
