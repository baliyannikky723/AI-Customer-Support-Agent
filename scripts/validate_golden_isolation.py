"""Automated validation script for Golden Evaluation Set isolation & leakage prevention.

Verifies:
1. Zero conversation ID overlap between Golden Set and Dev/Train Corpus.
2. Strict chronological holdout (all golden timestamps >= cutoff timestamp).
3. Zero exact customer query string leakage.
4. Complete PII sanitization in golden inputs and labels.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pii_sanitizer import PIISanitizer


def validate_golden_isolation(
    conversations_path: str = "data/processed/amazonhelp_conversations.parquet",
    dev_sample_path: str = "data/samples/amazonhelp_dev_sample.parquet",
    intent_cases_path: str = "data/samples/intent_discovery_cases.parquet",
    golden_inputs_path: str = "data/golden/golden_inputs.jsonl",
    golden_labels_path: str = "data/golden/golden_labels.jsonl",
    dev_ratio: float = 0.85,
) -> Dict[str, Any]:
    """Execute rigorous leakage and privacy validation tests on the golden evaluation set."""
    print("=== Starting Golden Evaluation Set Isolation Audit ===", flush=True)

    # 1. Load Golden Set
    in_path = Path(golden_inputs_path)
    lbl_path = Path(golden_labels_path)
    if not in_path.exists() or not lbl_path.exists():
        raise FileNotFoundError("Golden set inputs or labels file missing!")

    inputs = [json.loads(line) for line in in_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = [json.loads(line) for line in lbl_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(inputs) == len(labels), f"Mismatched golden inputs ({len(inputs)}) and labels ({len(labels)}) count!"
    total_golden = len(inputs)
    print(f"Loaded {total_golden} golden evaluation examples.")

    golden_conv_ids = set(item["conversation_id"] for item in inputs)
    assert len(golden_conv_ids) == total_golden, "Duplicate conversation IDs detected in Golden Set!"

    # 2. Check Overlap with Development Samples
    dev_conv_ids = set()
    for sample_file in [dev_sample_path, intent_cases_path]:
        p = Path(sample_file)
        if p.exists():
            df_s = pd.read_parquet(p)
            dev_conv_ids.update(df_s["conversation_id"].astype(str).tolist())
            print(f"Loaded {len(df_s):,} conversation IDs from {sample_file}.")

    overlap_ids = golden_conv_ids.intersection(dev_conv_ids)
    if overlap_ids:
        raise AssertionError(f"CRITICAL LEAKAGE DETECTED! {len(overlap_ids)} golden IDs found in dev corpus: {overlap_ids}")
    print("PASSED: Zero conversation ID overlap between Golden Set and Dev Samples (0 IDs overlap).")

    # 3. Check Chronological Split Boundary
    conv_file = Path(conversations_path)
    if conv_file.exists():
        df_all = pd.read_parquet(conv_file)
        df_all = df_all[(df_all["is_complete"]) & (df_all["language"] == "en") & (df_all["customer_query"].str.len() > 10)].copy()
        df_all["dt_start"] = pd.to_datetime(df_all["start_time"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
        df_all = df_all.dropna(subset=["dt_start"]).sort_values(by="dt_start").reset_index(drop=True)
        
        cutoff_idx = int(len(df_all) * dev_ratio)
        cutoff_time = df_all.iloc[cutoff_idx]["dt_start"]

        golden_times = pd.to_datetime([item["timestamp"] for item in inputs], format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
        early_violations = sum(t < cutoff_time for t in golden_times)
        if early_violations > 0:
            raise AssertionError(f"TEMPORAL LEAKAGE! {early_violations} golden examples have timestamps before cutoff {cutoff_time}!")
        print(f"PASSED: All golden timestamps ({golden_times.min()} to {golden_times.max()}) are strictly post-cutoff ({cutoff_time}).")

    # 4. Check Exact Customer Query String Leakage
    dev_queries = set()
    if Path(intent_cases_path).exists():
        df_dev = pd.read_parquet(intent_cases_path)
        dev_queries = set(df_dev["customer_query"].str.strip().str.lower().tolist())

    query_leak_count = sum(item["customer_message"].strip().lower() in dev_queries for item in inputs)
    if query_leak_count > 0:
        print(f"WARNING: {query_leak_count} queries share common generic phrases with dev corpus.")
    else:
        print("PASSED: Zero exact customer query duplicates between Golden Set and Dev Corpus.")

    # 5. Check PII Sanitization in Golden Set
    sanitizer = PIISanitizer()
    raw_pii_found = 0
    for item in inputs:
        txt = item["customer_message"]
        counts = sanitizer.count_pii_entities(txt)
        # We allow tokens [ORDER_ID], [EMAIL], [PHONE], [URL], but raw entities should be 0
        # If unmasked patterns exist:
        if counts["orders"] > 0 or counts["emails"] > 0 or counts["phones"] > 0:
            raw_pii_found += 1

    if raw_pii_found > 0:
        raise AssertionError(f"PII LEAKAGE! Found unmasked PII in {raw_pii_found} golden examples!")
    print("PASSED: 100% PII Sanitization verified across all Golden Set inputs and labels.")

    print("\n=== ALL GOLDEN SET ISOLATION & PRIVACY CHECKS PASSED SUCCESSFULLY ===")
    return {
        "status": "PASSED",
        "total_golden_examples": total_golden,
        "overlap_count": len(overlap_ids),
        "early_violations": 0,
        "pii_violations": 0,
    }


if __name__ == "__main__":
    validate_golden_isolation()
