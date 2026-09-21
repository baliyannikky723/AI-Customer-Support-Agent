"""Sample intent discovery cases from the Train/Dev partition of AmazonHelp conversations.

Ensures strict zero-leakage isolation by drawing samples exclusively from the
earlier 85% chronological development split with fixed seed=42.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sample_intent_cases(
    conversations_path: str = "data/processed/amazonhelp_conversations.parquet",
    output_path: str = "data/samples/intent_discovery_cases.parquet",
    sample_size: int = 8000,
    dev_split_ratio: float = 0.85,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Extract a representative, stratified sample from the train/dev partition for intent discovery."""
    np.random.seed(random_seed)
    conv_path = Path(conversations_path)
    if not conv_path.exists():
        raise FileNotFoundError(f"Conversations file not found: {conversations_path}")

    print(f"Loading conversations from {conversations_path}...", flush=True)
    df = pd.read_parquet(conversations_path)
    
    # Filter to English complete conversations with non-empty query
    df = df[(df["is_complete"]) & (df["language"] == "en") & (df["customer_query"].str.len() > 10)].copy()
    print(f"Found {len(df):,} valid complete English conversations.")

    # Chronological sort and partition (first 85% is Dev/Train)
    # Convert created_at format: "Wed Oct 11 09:14:31 +0000 2017"
    df["dt_start"] = pd.to_datetime(df["start_time"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
    df = df.sort_values(by="dt_start").reset_index(drop=True)
    
    cutoff_idx = int(len(df) * dev_split_ratio)
    dev_df = df.iloc[:cutoff_idx].copy()
    print(f"Development partition contains {len(dev_df):,} conversations (earlier {dev_split_ratio*100:.0f}% of timeline).")

    # Stratify by turn counts
    def turn_bucket(n):
        if n == 2:
            return "2_turns"
        elif n in (3, 4):
            return "3_4_turns"
        else:
            return "5_plus_turns"

    dev_df["turn_bucket"] = dev_df["num_turns"].apply(turn_bucket)

    target_per_bucket = sample_size // 3
    sampled_dfs = []

    for bucket, grp in dev_df.groupby("turn_bucket"):
        n_sample = min(len(grp), target_per_bucket)
        sampled_dfs.append(grp.sample(n=n_sample, random_state=random_seed))

    final_sample = pd.concat(sampled_dfs, ignore_index=True)
    
    # Fill remaining if needed
    if len(final_sample) < sample_size and len(dev_df) > len(final_sample):
        remaining = dev_df[~dev_df["conversation_id"].isin(final_sample["conversation_id"])]
        needed = sample_size - len(final_sample)
        if len(remaining) > 0:
            extra = remaining.sample(n=min(needed, len(remaining)), random_state=random_seed)
            final_sample = pd.concat([final_sample, extra], ignore_index=True)

    # Deterministic shuffle
    final_sample = final_sample.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    final_sample = final_sample.drop(columns=["dt_start", "turn_bucket"], errors="ignore")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    final_sample.to_parquet(out_file, index=False)

    print(f"Exported {len(final_sample):,} intent discovery cases to {output_path}.")
    return final_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample intent discovery cases.")
    parser.add_argument("--conversations-path", type=str, default="data/processed/amazonhelp_conversations.parquet")
    parser.add_argument("--output-path", type=str, default="data/samples/intent_discovery_cases.parquet")
    parser.add_argument("--sample-size", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    sample_intent_cases(
        conversations_path=args.conversations_path,
        output_path=args.output_path,
        sample_size=args.sample_size,
        random_seed=args.seed,
    )
