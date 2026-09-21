"""Create a stratified, diverse development sample from AmazonHelp conversations.

Produces a manageable parquet corpus (e.g. 5,000 conversations) stratified across
turn lengths, languages, and time periods for rapid local testing and development.
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


def create_dev_sample(
    conversations_path: str = "data/processed/amazonhelp_conversations.parquet",
    output_sample_path: str = "data/samples/amazonhelp_dev_sample.parquet",
    sample_size: int = 5000,
    language_mode: str = "english_only",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Create a stratified development sample from processed conversations."""
    np.random.seed(random_seed)
    conv_path = Path(conversations_path)
    if not conv_path.exists():
        raise FileNotFoundError(f"Processed conversations parquet not found at: {conversations_path}")

    print(f"Loading conversations from {conversations_path}...", flush=True)
    df = pd.read_parquet(conversations_path)
    print(f"Loaded {len(df):,} total conversations.", flush=True)

    # Filter to complete conversations with non-empty queries
    df = df[df["is_complete"] & (df["customer_query"].str.len() > 10)].copy()

    # Apply language filter if configured
    if language_mode == "english_only":
        df = df[df["language"] == "en"].copy()
        print(f"Filtered to English conversations: {len(df):,} remaining.")

    # Chronological partition: Strictly restrict to first 85% Development split
    df["dt_start"] = pd.to_datetime(df["start_time"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
    df = df.dropna(subset=["dt_start"]).sort_values(by="dt_start").reset_index(drop=True)
    cutoff_idx = int(len(df) * 0.85)
    df = df.iloc[:cutoff_idx].copy()
    print(f"Restricted development sampling pool to first 85% chronological split ({len(df):,} conversations).")

    # Stratify by turn counts (2 turns vs 3-4 turns vs 5+ turns)
    def turn_bucket(n):
        if n == 2:
            return "2_turns"
        elif n in (3, 4):
            return "3_4_turns"
        else:
            return "5_plus_turns"

    df["turn_bucket"] = df["num_turns"].apply(turn_bucket)

    # Stratified sampling across turn buckets
    target_per_bucket = sample_size // 3
    sampled_dfs = []

    for bucket, grp in df.groupby("turn_bucket"):
        n_sample = min(len(grp), target_per_bucket)
        sampled_dfs.append(grp.sample(n=n_sample, random_state=random_seed))

    df_sample = pd.concat(sampled_dfs, ignore_index=True)
    
    # If we need more to reach exact sample_size
    if len(df_sample) < sample_size and len(df) > len(df_sample):
        remaining = df[~df["conversation_id"].isin(df_sample["conversation_id"])]
        needed = sample_size - len(df_sample)
        if len(remaining) > 0:
            extra = remaining.sample(n=min(needed, len(remaining)), random_state=random_seed)
            df_sample = pd.concat([df_sample, extra], ignore_index=True)

    # Shuffle deterministically
    df_sample = df_sample.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    df_sample = df_sample.drop(columns=["turn_bucket"], errors="ignore")

    out_path = Path(output_sample_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_parquet(out_path, index=False)

    print(f"Created development sample: {output_sample_path} ({len(df_sample):,} conversations).")
    print(f"Turn distribution in dev sample:\n{df_sample['num_turns'].value_counts().to_dict()}")
    return df_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create AmazonHelp development sample.")
    parser.add_argument("--conversations-path", type=str, default="data/processed/amazonhelp_conversations.parquet")
    parser.add_argument("--output-path", type=str, default="data/samples/amazonhelp_dev_sample.parquet")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--language-mode", type=str, default="english_only", choices=["english_only", "all"])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    create_dev_sample(
        conversations_path=args.conversations_path,
        output_sample_path=args.output_path,
        sample_size=args.sample_size,
        language_mode=args.language_mode,
        random_seed=args.seed,
    )
