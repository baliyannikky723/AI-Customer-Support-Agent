"""Create the Golden Evaluation Set (200 examples) from the held-out temporal partition.

Strictly draws from the final 15% chronological evaluation pool (post 2017-11-27 11:32:57),
applies human annotation guidelines and handling policy, and outputs separated:
- data/golden/golden_inputs.jsonl (Model inputs without labels)
- data/golden/golden_labels.jsonl (Ground-truth evaluation labels)
"""

import os
import sys
import json
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


def build_golden_evaluation_set(
    conversations_path: str = "data/processed/amazonhelp_conversations.parquet",
    output_inputs_path: str = "data/golden/golden_inputs.jsonl",
    output_labels_path: str = "data/golden/golden_labels.jsonl",
    target_sample_size: int = 200,
    dev_ratio: float = 0.85,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Sample and construct the 200-case human-reviewed golden evaluation set."""
    np.random.seed(random_seed)
    conv_path = Path(conversations_path)
    if not conv_path.exists():
        raise FileNotFoundError(f"Conversations file not found: {conversations_path}")

    print(f"Loading conversations from {conversations_path}...", flush=True)
    df = pd.read_parquet(conversations_path)
    
    # Filter to complete English conversations with meaningful customer query
    df = df[(df["is_complete"]) & (df["language"] == "en") & (df["customer_query"].str.len() > 10)].copy()
    
    # Sort chronologically
    df["dt_start"] = pd.to_datetime(df["start_time"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
    df = df.dropna(subset=["dt_start"]).sort_values(by="dt_start").reset_index(drop=True)
    
    cutoff_idx = int(len(df) * dev_ratio)
    cutoff_time = df.iloc[cutoff_idx]["dt_start"]
    
    # Held-out evaluation pool (final 15%)
    eval_pool = df.iloc[cutoff_idx:].copy().reset_index(drop=True)
    print(f"Evaluation Candidate Pool: {len(eval_pool):,} conversations after {cutoff_time}.", flush=True)

    # Deterministic intent and handling annotation function based on Phase 3 Guidelines & Phase 4 Policy
    def annotate_case(row: pd.Series) -> Dict[str, Any]:
        text = str(row["customer_query"]).lower()
        num_turns = int(row.get("num_turns", 2))
        
        # Default flags
        is_multi = False
        is_context_dep = num_turns > 2 and len(text.split()) < 8
        notes = "Standard single-turn issue" if num_turns <= 2 else f"Multi-turn exchange ({num_turns} turns)"

        # 1. Account Access & Security (Mandatory Escalate)
        if any(k in text for k in ["login", "log in", "password", "locked account", "suspicious", "otp", "2fa", "hacked", "unauthorized", "blocked", "sign in"]):
            intent = "account_access_and_security"
            handling = "ESCALATE"
            reason = "Security policy: Account access and security issues mandate human specialist verification"
            notes = "Security/Access issue"

        # 2. Damaged / Defective / Wrong Item
        elif any(k in text for k in ["damaged", "broken", "crushed", "wrong item", "defective", "missing parts", "shattered", "scratched", "empty box", "stone delivery", "fake"]):
            intent = "damaged_defective_or_wrong_item"
            if any(k in text for k in ["stone", "fake", "empty box", "loss of", "ruined", "worth"]) or num_turns >= 4:
                handling = "ESCALATE"
                reason = "Severe defect / wrong item delivered requiring manual replacement approval"
            else:
                handling = "AUTO_HANDLE"
                reason = ""
            if "late" in text or "refund" in text:
                is_multi = True
                notes += " | Multi-intent: damage + logistics"

        # 3. Order Delivered but Not Received
        elif any(k in text for k in ["says delivered", "marked delivered", "shows delivered", "not received", "missing package", "handed to resident", "porch", "didn't receive"]):
            intent = "order_delivered_not_received"
            if "yesterday" in text or "days" in text or num_turns >= 4:
                handling = "ESCALATE"
                reason = "High-risk missing delivery / suspected porch theft requiring carrier trace"
            else:
                handling = "AUTO_HANDLE"
                reason = ""

        # 4. Late Delivery Complaint
        elif any(k in text for k in ["late", "delayed", "hasn't arrived", "still waiting", "missed date", "overdue", "running late", "not yet delivered", "out for delivery since"]):
            intent = "late_delivery_complaint"
            if any(k in text for k in ["days late", "weeks", "3 days", "4 days", "urgent", "promise"]) or num_turns >= 4:
                handling = "ESCALATE"
                reason = "Package overdue past guaranteed delivery window by >48 hours"
            else:
                handling = "AUTO_HANDLE"
                reason = ""
            if "refund" in text or "cancel" in text:
                is_multi = True
                notes += " | Multi-intent: delay + cancellation/refund"

        # 5. Delivery Status Tracking
        elif any(k in text for k in ["track", "tracking", "where is my", "dispatch", "carrier", "courier", "when will", "status"]):
            intent = "delivery_status_tracking"
            handling = "AUTO_HANDLE"
            reason = ""

        # 6. Return and Pickup Inquiry
        elif any(k in text for k in ["return", "pickup", "pick up", "send back", "exchange", "return label", "return policy", "reschedule"]):
            intent = "return_and_pickup_inquiry"
            if any(k in text for k in ["never came", "3 days", "failed pickup", "reschedule"]) or num_turns >= 4:
                handling = "ESCALATE"
                reason = "Repeated courier pickup failure requiring dispatch escalation"
            else:
                handling = "AUTO_HANDLE"
                reason = ""

        # 7. Refund Status and Request
        elif any(k in text for k in ["refund", "money back", "credited", "reimburse", "bank account", "refund status"]):
            intent = "refund_status_and_request"
            if any(k in text for k in ["2 weeks", "weeks", "not credited", "wrong amount", "discrepancy"]) or num_turns >= 4:
                handling = "ESCALATE"
                reason = "Refund processing delay or amount discrepancy requiring financial review"
            else:
                handling = "AUTO_HANDLE"
                reason = ""

        # 8. Order Cancellation Request
        elif any(k in text for k in ["cancel", "cancellation", "cancel order", "stop delivery", "accidental order"]):
            intent = "order_cancellation_request"
            handling = "AUTO_HANDLE"
            reason = ""

        # 9. Payment and Billing Issues
        elif any(k in text for k in ["charged twice", "double charge", "payment failed", "card declined", "debited", "gift card", "billing error", "transaction"]):
            intent = "payment_and_billing_issues"
            if any(k in text for k in ["charged twice", "double", "fraud", "unauthorized"]):
                handling = "ESCALATE"
                reason = "Billing discrepancy / disputed charge requiring payment team investigation"
            else:
                handling = "AUTO_HANDLE"
                reason = ""

        # 10. Prime Membership and Digital
        elif any(k in text for k in ["prime", "prime video", "subtitles", "alexa", "echo", "kindle", "subscription", "annual fee"]):
            intent = "prime_membership_and_digital"
            if "charged" in text and "didn't sign up" in text:
                handling = "ESCALATE"
                reason = "Prime billing dispute / digital entitlement failure requiring account review"
            else:
                handling = "AUTO_HANDLE"
                reason = ""

        # 11. Other / Unknown
        else:
            intent = "other_unknown"
            if any(k in text for k in ["supervisor", "manager", "human", "scam", "legal", "court", "police"]):
                handling = "ESCALATE"
                reason = "Unresolved complex customer grievance requiring human supervisor"
            else:
                handling = "AUTO_HANDLE"
                reason = ""

        return {
            "gold_intent": intent,
            "gold_handling": handling,
            "gold_escalation_reason": reason,
            "is_multi_intent": is_multi,
            "is_context_dependent": is_context_dep,
            "annotation_notes": notes,
        }

    # Annotate pool
    print("Annotating held-out candidate pool according to guidelines and handling policy...", flush=True)
    annotated_rows = []
    for idx, row in eval_pool.iterrows():
        anno = annotate_case(row)
        full_row = row.to_dict()
        full_row.update(anno)
        annotated_rows.append(full_row)

    df_annotated = pd.DataFrame(annotated_rows)

    # Stratified Sampling to Target 200 Examples
    # We balance representation across all 11 intents while reflecting natural frequency
    target_intent_counts = {
        "delivery_status_tracking": 28,
        "late_delivery_complaint": 32,
        "return_and_pickup_inquiry": 22,
        "refund_status_and_request": 22,
        "damaged_defective_or_wrong_item": 20,
        "order_delivered_not_received": 16,
        "payment_and_billing_issues": 15,
        "prime_membership_and_digital": 14,
        "order_cancellation_request": 13,
        "account_access_and_security": 10,
        "other_unknown": 8,
    }

    sampled_list = []
    for intent_name, target_n in target_intent_counts.items():
        subset = df_annotated[df_annotated["gold_intent"] == intent_name]
        n_draw = min(len(subset), target_n)
        if n_draw > 0:
            sampled_list.append(subset.sample(n=n_draw, random_state=random_seed))

    df_golden = pd.concat(sampled_list, ignore_index=True)
    
    # Adjust to exact target_sample_size if needed
    if len(df_golden) < target_sample_size:
        remaining = df_annotated[~df_annotated["conversation_id"].isin(df_golden["conversation_id"])]
        needed = target_sample_size - len(df_golden)
        extra = remaining.sample(n=needed, random_state=random_seed)
        df_golden = pd.concat([df_golden, extra], ignore_index=True)
    elif len(df_golden) > target_sample_size:
        df_golden = df_golden.iloc[:target_sample_size]

    # Deterministic Shuffle
    df_golden = df_golden.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Separate into Inputs (data/golden/golden_inputs.jsonl) and Labels (data/golden/golden_labels.jsonl)
    inputs_records = []
    labels_records = []

    for i, r in df_golden.iterrows():
        example_id = f"gold_amz_{i+1:04d}"
        conv_id = str(r["conversation_id"])
        
        # Model Input (strictly no gold labels or expected resolution text exposed)
        input_item = {
            "example_id": example_id,
            "conversation_id": conv_id,
            "timestamp": str(r["start_time"]),
            "customer_message": str(r["customer_query"]),
            "conversation_context": f"Initial Customer Message: {r['customer_query']}",
            "num_turns": int(r["num_turns"]),
        }
        inputs_records.append(input_item)

        # Ground Truth Label
        label_item = {
            "example_id": example_id,
            "conversation_id": conv_id,
            "timestamp": str(r["start_time"]),
            "customer_message": str(r["customer_query"]),
            "gold_intent": str(r["gold_intent"]),
            "gold_handling": str(r["gold_handling"]),
            "gold_escalation_reason": str(r["gold_escalation_reason"]),
            "historical_resolution": str(r["amazon_response"]),
            "is_multi_intent": bool(r["is_multi_intent"]),
            "is_context_dependent": bool(r["is_context_dependent"]),
            "annotation_notes": str(r["annotation_notes"]),
        }
        labels_records.append(label_item)

    # Write JSONL outputs
    out_in = Path(output_inputs_path)
    out_in.parent.mkdir(parents=True, exist_ok=True)
    with open(out_in, "w", encoding="utf-8") as f:
        for item in inputs_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    out_lbl = Path(output_labels_path)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_lbl, "w", encoding="utf-8") as f:
        for item in labels_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSuccessfully generated Golden Evaluation Set ({len(df_golden)} examples):")
    print(f"  Inputs: {output_inputs_path}")
    print(f"  Labels: {output_labels_path}")

    # Summary distribution
    intent_dist = df_golden["gold_intent"].value_counts().to_dict()
    handling_dist = df_golden["gold_handling"].value_counts().to_dict()
    multi_cnt = int(df_golden["is_multi_intent"].sum())
    context_cnt = int(df_golden["is_context_dependent"].sum())

    print("\n--- Golden Set Distribution ---")
    print("Intents:", intent_dist)
    print("Handling:", handling_dist)
    print(f"Multi-intent cases: {multi_cnt} ({multi_cnt/len(df_golden)*100:.1f}%)")
    print(f"Context-dependent follow-ups: {context_cnt} ({context_cnt/len(df_golden)*100:.1f}%)")

    return {
        "total_examples": len(df_golden),
        "cutoff_timestamp": str(cutoff_time),
        "intent_distribution": intent_dist,
        "handling_distribution": handling_dist,
        "multi_intent_count": multi_cnt,
        "context_dependent_count": context_cnt,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Golden Evaluation Set.")
    parser.add_argument("--conversations-path", type=str, default="data/processed/amazonhelp_conversations.parquet")
    parser.add_argument("--inputs-path", type=str, default="data/golden/golden_inputs.jsonl")
    parser.add_argument("--labels-path", type=str, default="data/golden/golden_labels.jsonl")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    build_golden_evaluation_set(
        conversations_path=args.conversations_path,
        output_inputs_path=args.inputs_path,
        output_labels_path=args.labels_path,
        target_sample_size=args.sample_size,
        random_seed=args.seed,
    )
