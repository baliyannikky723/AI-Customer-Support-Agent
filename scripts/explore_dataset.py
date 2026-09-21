"""Comprehensive and fast Exploratory Data Analysis for Customer Support on Twitter dataset.

Uses vectorized operations and chunked reading to process all 2.81M records in seconds,
computing exact schema statistics, brand statistics, conversation linkages,
and data quality indicators without excessive memory consumption.
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np


def analyze_dataset(
    raw_data_path: str = "data/raw/twcs.csv",
    output_brand_stats_path: str = "data/processed/brand_statistics.csv",
    chunk_size: int = 500_000,
) -> dict:
    """Analyze the full dataset in chunks using vectorized operations and save brand statistics."""
    start_time = time.time()
    raw_path = Path(raw_data_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_data_path}")

    file_size_mb = raw_path.stat().st_size / (1024 * 1024)

    # Accumulators for overall metrics
    total_rows = 0
    missing_counts = defaultdict(int)
    inbound_counts = Counter()
    brand_outbound_counts = Counter()
    brand_responded_parents = defaultdict(set)
    brand_outbound_tweet_ids = defaultdict(set)
    
    empty_text_count = 0
    short_text_count = 0
    long_text_count = 0
    
    in_response_to_not_null = 0
    response_tweet_id_not_null = 0
    multi_response_count = 0

    print(f"Pass 1: Scanning {raw_data_path} ({file_size_mb:.2f} MB)...", flush=True)

    # Pass 1: Global metrics & Outbound brand indexing
    for chunk in pd.read_csv(
        raw_data_path,
        chunksize=chunk_size,
        dtype={
            "tweet_id": "int64",
            "author_id": "str",
            "inbound": "bool",
            "created_at": "str",
            "text": "str",
            "response_tweet_id": "str",
            "in_response_to_tweet_id": "float64",
        },
        low_memory=False,
    ):
        num_rows = len(chunk)
        total_rows += num_rows

        # Missing counts
        for col in chunk.columns:
            missing_counts[col] += int(chunk[col].isna().sum())

        # Inbound vs Outbound
        inb_vc = chunk["inbound"].value_counts()
        for k, v in inb_vc.items():
            inbound_counts[bool(k)] += int(v)

        # Text quality metrics
        text_series = chunk["text"].fillna("")
        text_lens = text_series.str.len()
        empty_text_count += int((text_lens == 0).sum())
        short_text_count += int(((text_lens > 0) & (text_lens < 10)).sum())
        long_text_count += int((text_lens > 200).sum())

        # Threading links
        in_response_to_not_null += int(chunk["in_response_to_tweet_id"].notna().sum())
        resp_series = chunk["response_tweet_id"].dropna()
        response_tweet_id_not_null += len(resp_series)
        multi_response_count += int(resp_series.str.contains(",").sum())

        # Outbound brand tweets analysis
        outbounds = chunk[chunk["inbound"] == False]
        if not outbounds.empty:
            outbound_vc = outbounds["author_id"].value_counts()
            for author, cnt in outbound_vc.items():
                brand_outbound_counts[str(author)] += int(cnt)

            # Record parent tweet IDs that brand replied to
            valid_parents = outbounds.dropna(subset=["in_response_to_tweet_id"])
            for author, grp in valid_parents.groupby("author_id"):
                brand_responded_parents[str(author)].update(grp["in_response_to_tweet_id"].astype(int))
                brand_outbound_tweet_ids[str(author)].update(grp["tweet_id"].astype(int))

        print(f"  Processed {total_rows:,} rows...", flush=True)

    pass1_time = time.time() - start_time
    print(f"Pass 1 complete in {pass1_time:.2f}s ({total_rows:,} rows total).", flush=True)

    # Identify all active brand accounts (outbound count >= 10)
    all_brands = [b for b, cnt in brand_outbound_counts.most_common() if cnt >= 10]
    print(f"Found {len(all_brands)} active brands in the dataset.", flush=True)

    # Pass 2: Count customer incoming inquiries per brand (inbound tweets mentioning @brand or in response to brand)
    print("Pass 2: Analyzing customer inquiries per brand...", flush=True)
    
    # Precompute fast lookup sets for top brands
    brand_mentions = {b: f"@{b.lower()}" for b in all_brands[:40]}
    brand_cust_counts = defaultdict(int)

    # Flatten brand outbound tweet IDs to a mapping of tweet_id -> brand for customer follow-up matching
    parent_to_brand = {}
    for b, tids in brand_outbound_tweet_ids.items():
        for tid in tids:
            parent_to_brand[tid] = b

    for chunk in pd.read_csv(
        raw_data_path,
        chunksize=chunk_size,
        usecols=["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"],
        dtype={
            "tweet_id": "int64",
            "author_id": "str",
            "inbound": "bool",
            "text": "str",
            "in_response_to_tweet_id": "float64",
        },
        low_memory=False,
    ):
        inbounds = chunk[chunk["inbound"] == True]
        if inbounds.empty:
            continue

        # Check follow-ups (in_response_to_tweet_id pointing to a brand tweet)
        inb_with_parent = inbounds.dropna(subset=["in_response_to_tweet_id"])
        if not inb_with_parent.empty:
            parent_ids = inb_with_parent["in_response_to_tweet_id"].astype(int)
            for pid in parent_ids:
                if pid in parent_to_brand:
                    brand_cust_counts[parent_to_brand[pid]] += 1

        # Check initial inquiries mentioning brand
        text_lower = inbounds["text"].str.lower().fillna("")
        for b, handle in brand_mentions.items():
            matches = int(text_lower.str.contains(handle, regex=False).sum())
            brand_cust_counts[b] += matches

    # Assemble brand statistics DataFrame
    records = []
    for b in all_brands:
        support_cnt = brand_outbound_counts[b]
        responded_convs = len(brand_responded_parents[b])
        # Customer messages is at least responded_convs plus any followups/mentions
        cust_cnt = max(brand_cust_counts[b], responded_convs)
        total_msgs = support_cnt + cust_cnt
        est_convs = max(cust_cnt, responded_convs)
        usable_convs = responded_convs

        records.append({
            "brand": b,
            "total_messages": total_msgs,
            "customer_messages": cust_cnt,
            "support_messages": support_cnt,
            "estimated_conversations": est_convs,
            "responded_conversations": responded_convs,
            "usable_conversations": usable_convs,
        })

    brand_stats_df = pd.DataFrame(records)
    brand_stats_df = brand_stats_df.sort_values(by="usable_conversations", ascending=False).reset_index(drop=True)

    # Save to CSV
    output_brand_csv = Path(output_brand_stats_path)
    output_brand_csv.parent.mkdir(parents=True, exist_ok=True)
    brand_stats_df.to_csv(output_brand_csv, index=False)
    print(f"Saved brand statistics to {output_brand_stats_path}")

    total_time = time.time() - start_time
    print(f"All analysis completed in {total_time:.2f}s.")

    summary = {
        "file_size_mb": round(file_size_mb, 2),
        "total_rows": total_rows,
        "columns": list(missing_counts.keys()),
        "missing_counts": dict(missing_counts),
        "missing_percentages": {col: round(cnt / total_rows * 100, 2) for col, cnt in missing_counts.items()},
        "inbound_count": inbound_counts[True],
        "outbound_count": inbound_counts[False],
        "unique_brands_count": len(all_brands),
        "empty_text_count": empty_text_count,
        "short_text_count": short_text_count,
        "long_text_count": long_text_count,
        "in_response_to_not_null": in_response_to_not_null,
        "response_tweet_id_not_null": response_tweet_id_not_null,
        "multi_response_count": multi_response_count,
        "top_10_brands": brand_stats_df.head(10).to_dict(orient="records"),
    }
    return summary


if __name__ == "__main__":
    summary = analyze_dataset()
    print("\n================ DATASET ANALYSIS SUMMARY ================")
    print(f"Total Rows / Tweets: {summary['total_rows']:,}")
    print(f"Inbound (Customer) Tweets: {summary['inbound_count']:,} ({summary['inbound_count']/summary['total_rows']*100:.2f}%)")
    print(f"Outbound (Brand/Agent) Tweets: {summary['outbound_count']:,} ({summary['outbound_count']/summary['total_rows']*100:.2f}%)")
    print(f"Total Active Brands: {summary['unique_brands_count']}")
    print(f"Missing in_response_to_tweet_id: {summary['missing_percentages']['in_response_to_tweet_id']}% (Root tweets)")
    print(f"Missing response_tweet_id: {summary['missing_percentages']['response_tweet_id']}% (Terminal tweets)")
    print("\nTop 10 Brands by Usable Support Conversations:")
    for i, b in enumerate(summary["top_10_brands"], 1):
        print(f"  {i}. {b['brand']:<16} | Usable Convs: {b['usable_conversations']:,} | Support Msgs: {b['support_messages']:,} | Cust Msgs: {b['customer_messages']:,}")
