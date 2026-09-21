"""Reconstruct multi-turn customer support conversations for AmazonHelp.

Groups messages into conversation threads, sanitizes PII, detects language,
identifies atomic (customer_query, amazon_response) pairs, and outputs
data/processed/amazonhelp_conversations.parquet.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pii_sanitizer import PIISanitizer
from src.preprocessing.language_detector import LanguageDetector


def build_conversations(
    raw_parquet_path: str = "data/processed/amazonhelp_raw.parquet",
    output_conversations_path: str = "data/processed/amazonhelp_conversations.parquet",
    language_mode: str = "all",
    sanitize_pii: bool = True,
) -> pd.DataFrame:
    """Reconstruct complete conversation threads from AmazonHelp raw parquet dataset."""
    start_time = time.time()
    raw_path = Path(raw_parquet_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Input parquet file not found at: {raw_parquet_path}")

    print(f"Loading AmazonHelp corpus from {raw_parquet_path}...", flush=True)
    df = pd.read_parquet(raw_parquet_path)
    print(f"Loaded {len(df):,} messages. Building conversation graph...", flush=True)

    sanitizer = PIISanitizer()
    lang_detector = LanguageDetector()

    # Pre-index messages by tweet_id for fast $O(1)$ lookups
    # Convert dataframe to record dictionary indexed by tweet_id
    records_by_id = {}
    for r in df.to_dict(orient="records"):
        tid = int(r["tweet_id"])
        records_by_id[tid] = r

    # Build adjacency maps
    parent_map: Dict[int, int] = {}  # child_id -> parent_id
    children_map: Dict[int, List[int]] = {}  # parent_id -> list of child_ids

    for tid, r in records_by_id.items():
        parent_id = r.get("in_response_to_tweet_id")
        if pd.notna(parent_id):
            pid = int(parent_id)
            parent_map[tid] = pid
            if pid not in children_map:
                children_map[pid] = []
            children_map[pid].append(tid)

    # Find Root Nodes for AmazonHelp conversations
    # A conversation thread root is any customer tweet that was either:
    # 1. Directly replied to by AmazonHelp, or
    # 2. Has no parent in the dataset (or parent not in records)
    outbound_amazon_tids = [tid for tid, r in records_by_id.items() if not r["inbound"] and r["author_id"] == "AmazonHelp"]
    
    print(f"Found {len(outbound_amazon_tids):,} AmazonHelp support tweets. Tracing thread trees...", flush=True)

    # Find all distinct roots that lead to AmazonHelp interactions
    visited_tweets = set()
    conversation_trees = []

    def find_root(tid: int) -> int:
        curr = tid
        while curr in parent_map and parent_map[curr] in records_by_id:
            curr = parent_map[curr]
        return curr

    # Collect unique root IDs
    root_ids = set()
    for tid in outbound_amazon_tids:
        root_id = find_root(tid)
        root_ids.add(root_id)

    print(f"Identified {len(root_ids):,} distinct conversation root tweets. Reconstructing threads...", flush=True)

    # Reconstruct each tree via BFS / chronological sort
    conversations = []
    
    for root_id in root_ids:
        # Collect all nodes in this conversation tree
        tree_nodes = []
        queue = [root_id]
        tree_node_ids = set()

        while queue:
            curr_id = queue.pop(0)
            if curr_id in tree_node_ids:
                continue
            tree_node_ids.add(curr_id)
            if curr_id in records_by_id:
                tree_nodes.append(records_by_id[curr_id])
            for child_id in children_map.get(curr_id, []):
                if child_id not in tree_node_ids:
                    queue.append(child_id)

        if not tree_nodes:
            continue

        # Sort messages in the thread chronologically
        # Note: created_at format is "Wed Oct 11 09:14:31 +0000 2017"
        # We can sort using pd.to_datetime or preserved message sequence
        tree_nodes_sorted = sorted(tree_nodes, key=lambda x: str(x.get("created_at", "")))

        # Extract turns
        turns = []
        cust_texts = []
        supp_texts = []
        sanitized_cust_texts = []
        sanitized_supp_texts = []

        for node in tree_nodes_sorted:
            raw_text = str(node.get("text", ""))
            san_text = sanitizer.sanitize(raw_text) if sanitize_pii else raw_text
            is_inb = bool(node.get("inbound", False))
            author = str(node.get("author_id", ""))
            
            turn_info = {
                "tweet_id": int(node["tweet_id"]),
                "author_id": author,
                "inbound": is_inb,
                "created_at": str(node.get("created_at", "")),
                "raw_text": raw_text,
                "sanitized_text": san_text,
            }
            turns.append(turn_info)

            if is_inb:
                cust_texts.append(raw_text)
                sanitized_cust_texts.append(san_text)
            else:
                supp_texts.append(raw_text)
                sanitized_supp_texts.append(san_text)

        has_cust = len(cust_texts) > 0
        has_supp = len(supp_texts) > 0
        is_complete = has_cust and has_supp

        # Primary customer inquiry (first customer message in thread)
        first_cust_raw = cust_texts[0] if cust_texts else ""
        first_cust_san = sanitized_cust_texts[0] if sanitized_cust_texts else ""
        
        # Primary Amazon response (first support reply in thread)
        first_supp_raw = supp_texts[0] if supp_texts else ""
        first_supp_san = sanitized_supp_texts[0] if sanitized_supp_texts else ""
        
        # Complete resolution text (all support replies joined)
        all_resolution_san = " \n ".join(sanitized_supp_texts)

        # Detect language of initial customer inquiry
        lang, conf = lang_detector.detect(first_cust_raw) if first_cust_raw else ("ambiguous", 0.0)

        # Filter by language mode if requested
        if language_mode == "english_only" and lang != "en":
            continue

        start_time_str = tree_nodes_sorted[0].get("created_at", "")
        end_time_str = tree_nodes_sorted[-1].get("created_at", "")

        conversations.append({
            "conversation_id": str(root_id),
            "message_count": len(tree_nodes_sorted),
            "customer_message_count": len(cust_texts),
            "support_message_count": len(supp_texts),
            "is_complete": is_complete,
            "has_customer_message": has_cust,
            "has_support_response": has_supp,
            "language": lang,
            "language_confidence": round(conf, 3),
            "start_time": str(start_time_str),
            "end_time": str(end_time_str),
            "customer_query": first_cust_san,
            "customer_query_raw": first_cust_raw,
            "amazon_response": first_supp_san,
            "amazon_response_raw": first_supp_raw,
            "resolution_text": all_resolution_san,
            "num_turns": len(turns),
        })

    df_convs = pd.DataFrame(conversations)
    print(f"Reconstructed {len(df_convs):,} conversations.")

    # Export to Parquet
    out_parquet = Path(output_conversations_path)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_convs.to_parquet(out_parquet, index=False)
    
    elapsed = time.time() - start_time
    print(f"Saved conversations to {output_conversations_path} in {elapsed:.2f}s.")

    # Print summary metrics
    complete_cnt = int(df_convs["is_complete"].sum())
    incomplete_cnt = len(df_convs) - complete_cnt
    usable_cases_cnt = int(((df_convs["is_complete"]) & (df_convs["customer_query"].str.len() > 10)).sum())

    print("\n--- Conversation Reconstruction Summary ---")
    print(f"Total Conversations: {len(df_convs):,}")
    print(f"Complete Conversations (Cust + Supp): {complete_cnt:,} ({complete_cnt/len(df_convs)*100:.2f}%)")
    print(f"Incomplete Conversations: {incomplete_cnt:,} ({incomplete_cnt/len(df_convs)*100:.2f}%)")
    print(f"Usable Historical Cases (Complete & meaningful query): {usable_cases_cnt:,}")
    print("\nTurn Count Distribution:")
    print(df_convs["num_turns"].value_counts().head(5).to_dict())

    return df_convs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build AmazonHelp conversation threads.")
    parser.add_argument("--raw-parquet", type=str, default="data/processed/amazonhelp_raw.parquet", help="Path to raw AmazonHelp parquet")
    parser.add_argument("--output-parquet", type=str, default="data/processed/amazonhelp_conversations.parquet", help="Path to output conversations parquet")
    parser.add_argument("--language-mode", type=str, default="all", choices=["all", "english_only"], help="Language filtering mode")
    parser.add_argument("--sanitize-pii", action="store_true", default=True, help="Enable PII sanitization")

    args = parser.parse_args()
    build_conversations(
        raw_parquet_path=args.raw_parquet,
        output_conversations_path=args.output_parquet,
        language_mode=args.language_mode,
        sanitize_pii=args.sanitize_pii,
    )
