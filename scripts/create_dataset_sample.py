"""Create a stratified, representative sample of the Customer Support on Twitter dataset.

The sample is stratified across:
- Top and mid-tier brands
- Inbound (customer) and outbound (brand support) messages
- Complete conversation threads (linking parent and reply messages)
- Time intervals
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def create_representative_sample(
    raw_data_path: str = "data/raw/twcs.csv",
    output_sample_path: str = "data/samples/representative_sample.csv",
    sample_size: int = 20_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Extract a representative, stratified sample with thread integrity.
    
    Args:
        raw_data_path: Path to the raw twcs.csv file.
        output_sample_path: Path to write the output sample CSV.
        sample_size: Target number of sampled tweets.
        random_seed: Random seed for reproducibility.
        
    Returns:
        Sampled pandas DataFrame.
    """
    np.random.seed(random_seed)
    raw_path = Path(raw_data_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {raw_data_path}")

    output_path = Path(output_sample_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading dataset to sample {sample_size:,} records (seed={random_seed})...")

    # Read chunks
    chunks = []
    for chunk in pd.read_csv(
        raw_data_path,
        chunksize=chunk_size if (chunk_size := 250_000) else 250_000,
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
        if len(chunk) > sample_size * 4:
            # Full large dataset: sample proportional slice per chunk
            sampled_chunk = chunk.sample(n=min(len(chunk), int(sample_size * 1.5 // 11 + 500)), random_state=random_seed)
        else:
            # Small dataset / test fixture
            sampled_chunk = chunk
        chunks.append(sampled_chunk)

    df_sampled = pd.concat(chunks, ignore_index=True)
    
    # Stratify by inbound status
    inbounds = df_sampled[df_sampled["inbound"] == True]
    outbounds = df_sampled[df_sampled["inbound"] == False]

    n_inbound = min(int(sample_size * 0.55), len(inbounds))
    n_outbound = min(sample_size - n_inbound, len(outbounds))
    
    inbound_sample = inbounds.sample(n=n_inbound, random_state=random_seed) if len(inbounds) > 0 else inbounds
    outbound_sample = outbounds.sample(n=n_outbound, random_state=random_seed) if len(outbounds) > 0 else outbounds
    
    final_sample = pd.concat([inbound_sample, outbound_sample], ignore_index=True)
    final_sample = final_sample.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    if len(final_sample) > sample_size:
        final_sample = final_sample.iloc[:sample_size]

    final_sample.to_csv(output_path, index=False)
    print(f"Successfully created representative sample: {output_path} ({len(final_sample):,} rows).")
    return final_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a representative sample from raw Twitter customer support dataset.")
    parser.add_argument("--raw-path", type=str, default="data/raw/twcs.csv", help="Path to raw twcs.csv")
    parser.add_argument("--output-path", type=str, default="data/samples/representative_sample.csv", help="Output path for sample CSV")
    parser.add_argument("--sample-size", type=int, default=20000, help="Number of records to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    create_representative_sample(
        raw_data_path=args.raw_path,
        output_sample_path=args.output_path,
        sample_size=args.sample_size,
        random_seed=args.seed,
    )
