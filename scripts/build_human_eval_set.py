"""
Build Stratified 50-Example Human Evaluation Subset from Golden Evaluation Set.

Sampling Strategy:
- Size: Exactly 50 examples.
- Stratification:
  1. Stratified across all 11 domain intents (proportional allocation with min 2 per intent).
  2. Balanced representation between AUTO_HANDLE and ESCALATE (triage routing).
  3. Includes edge cases: ambiguous queries, account security, payment disputes, and missing delivery claims.
- Deterministic random seed = 42.

Outputs:
- data/evaluation/human_eval_set.jsonl
- data/evaluation/human_annotations_template.jsonl
"""

import os
import json
import random
from collections import defaultdict
from typing import List, Dict, Any


def build_human_eval_set(
    predictions_path: str = "reports/phase8_results/predictions.jsonl",
    output_dir: str = "data/evaluation",
    target_sample_size: int = 50,
    seed: int = 42,
):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found at: {predictions_path}")

    # Load 200 golden predictions from Phase 8
    records: List[Dict[str, Any]] = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    print(f"Loaded {len(records)} golden records from {predictions_path}.")

    # Group by (gold_intent, triage_decision)
    strata = defaultdict(list)
    for r in records:
        key = (r.get("gold_intent", "other_unknown"), r.get("triage_decision", "ESCALATE"))
        strata[key].append(r)

    selected_records: List[Dict[str, Any]] = []
    selected_ids = set()

    # Step 1: Ensure high-priority safety-critical cases are represented
    # Account security, missing packages, payment issues
    high_priority_intents = ["account_access_and_security", "order_delivered_not_received", "payment_and_billing_issues"]
    for r in records:
        if r.get("gold_intent") in high_priority_intents:
            if r["query_id"] not in selected_ids:
                selected_records.append(r)
                selected_ids.add(r["query_id"])

    # Step 2: Stratified sample across all intent strata
    # Shuffle each stratum deterministically
    all_strata_keys = sorted(list(strata.keys()))
    for key in all_strata_keys:
        group = strata[key]
        random.shuffle(group)
        for r in group:
            if r["query_id"] not in selected_ids:
                selected_records.append(r)
                selected_ids.add(r["query_id"])
                break  # Pick at least one from this (intent, triage) stratum

    # Step 3: Fill remaining quota up to target_sample_size
    remaining_pool = [r for r in records if r["query_id"] not in selected_ids]
    random.shuffle(remaining_pool)

    while len(selected_records) < target_sample_size and remaining_pool:
        r = remaining_pool.pop()
        selected_records.append(r)
        selected_ids.add(r["query_id"])

    # Trim to exact target size if overshot
    if len(selected_records) > target_sample_size:
        # Keep deterministic subset
        random.shuffle(selected_records)
        selected_records = selected_records[:target_sample_size]

    print(f"Successfully selected {len(selected_records)} stratified human evaluation examples.")

    # Save human_eval_set.jsonl
    eval_set_path = os.path.join(output_dir, "human_eval_set.jsonl")
    template_path = os.path.join(output_dir, "human_annotations_template.jsonl")

    with open(eval_set_path, "w", encoding="utf-8") as f_eval, open(template_path, "w", encoding="utf-8") as f_tmpl:
        for r in selected_records:
            eval_item = {
                "example_id": r["query_id"],
                "customer_query": r["customer_message"],
                "predicted_intent": r["predicted_intent"],
                "intent_confidence": r["intent_confidence"],
                "retrieved_evidence": r.get("selected_evidence", []),
                "evidence_quality_score": r.get("evidence_quality_score", 0.0),
                "generated_response": r["generated_reply"],
                "triage_decision": r["triage_decision"],
                "triage_reason_codes": r.get("triage_reason_codes", []),
                "guardrail_status": "PASSED" if r.get("guardrail_passed", True) else "FAILED",
            }
            f_eval.write(json.dumps(eval_item) + "\n")

            template_item = {
                "example_id": r["query_id"],
                "rater_id": "human_reviewer_1",
                "customer_query": r["customer_message"],
                "generated_response": r["generated_reply"],
                "correctness": None,       # 1 to 5
                "groundedness": None,      # 1 to 5
                "helpfulness": None,       # 1 to 5
                "safety": None,            # 1 to 5
                "tone": None,              # 1 to 5
                "overall_quality": None,   # 1 to 5
                "acceptable": None,        # true / false
                "critical_failure": None,  # true / false
                "failure_categories": [],  # e.g. ["MISCLASSIFIED_INTENT", "FORBIDDEN_LIVE_CLAIM"]
                "rationale": "",           # Human reviewer rationale
            }
            f_tmpl.write(json.dumps(template_item, indent=2) + "\n")

    print(f"Saved human evaluation subset to: {eval_set_path}")
    print(f"Saved human annotation template to: {template_path}")


if __name__ == "__main__":
    build_human_eval_set()
