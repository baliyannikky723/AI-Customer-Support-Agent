# Golden Evaluation Set Specification & Audit Report

This report documents the sampling strategy, schema separation, human-grounded annotation methodology, and leakage-prevention audit for the **200-case Amazon Customer Support Golden Evaluation Set**.

---

## 1. Executive Summary & Purpose

The Golden Evaluation Set serves as the **uncontaminated ground-truth benchmark** for all subsequent evaluation phases:
- **Intent Classification Accuracy**: Micro/Macro F1 across 11 defined intents.
- **Triage & Routing Fidelity**: Precision/Recall/F1 for `AUTO_HANDLE` vs. `ESCALATE`.
- **Grounded Reply Quality**: Evaluating factual correctness against retrieved policy without hallucination.
- **Human vs. LLM-as-a-Judge Agreement**: Computing Cohen's Kappa ($\kappa$) and Pearson correlation between automated judge scores and human rubrics.

---

## 2. Temporal Holdout & Isolation Architecture

To prevent data contamination and self-retrieval leakage:
- **Chronological Boundary**: Cutoff timestamp established at **`2017-11-27 11:32:57+00:00`** (85th percentile of complete English conversations).
- **Development & Retrieval Corpus**: 55,475 conversations ($85.0\%$) spanning `2015-06-13` to `2017-11-27 11:30:51`.
- **Evaluation Candidate Pool**: 9,790 conversations ($15.0\%$) spanning `2017-11-27 11:32:57` to `2017-12-03 23:04:17`.
- **Zero-Overlap Guarantee**: Verified via `scripts/validate_golden_isolation.py` ($\text{IDs}_{\text{golden}} \cap \text{IDs}_{\text{dev}} = \emptyset$).

---

## 3. Input & Ground-Truth Label Schema Separation

To prevent test-time leakage into the agent pipeline, model inputs and evaluation labels are stored in separate files linked by `example_id`:

### A. Model Input (`data/golden/golden_inputs.jsonl`)
```json
{
  "example_id": "gold_amz_0001",
  "conversation_id": "2913143",
  "timestamp": "Mon Nov 27 13:01:51 +0000 2017",
  "customer_message": "Where is my package? The tracking hasn't updated since yesterday.",
  "conversation_context": "Initial Customer Message: Where is my package? The tracking hasn't updated since yesterday.",
  "num_turns": 2
}
```

### B. Evaluation Ground Truth (`data/golden/golden_labels.jsonl`)
```json
{
  "example_id": "gold_amz_0001",
  "conversation_id": "2913143",
  "timestamp": "Mon Nov 27 13:01:51 +0000 2017",
  "customer_message": "Where is my package? The tracking hasn't updated since yesterday.",
  "gold_intent": "delivery_status_tracking",
  "gold_handling": "AUTO_HANDLE",
  "gold_escalation_reason": "",
  "historical_resolution": "[CUSTOMER] You can track the latest courier scans in Your Orders: [URL]",
  "is_multi_intent": false,
  "is_context_dependent": false,
  "annotation_notes": "Standard single-turn issue"
}
```

---

## 4. Golden Set Distribution & Composition

### A. Intent Class Distribution ($N=200$)

| Intent Name | Example Count | Percentage | Candidate Pool Approx. % | Representation Rationale |
|---|---|---|---|---|
| **`late_delivery_complaint`** | 32 | 16.0% | 16.2% | High natural frequency; key operational driver. |
| **`delivery_status_tracking`** | 28 | 14.0% | 14.5% | Core tracking benchmark. |
| **`return_and_pickup_inquiry`** | 22 | 11.0% | 11.2% | Reverse logistics & pickup rescheduling. |
| **`refund_status_and_request`** | 22 | 11.0% | 10.8% | Financial timeline & dispute inquiries. |
| **`damaged_defective_or_wrong_item`**| 20 | 10.0% | 9.4% | Physical product defect replacements. |
| **`order_delivered_not_received`** | 16 | 8.0% | 7.8% | High-risk missing package investigations. |
| **`payment_and_billing_issues`** | 15 | 7.5% | 7.1% | Double-charges, payment failures. |
| **`prime_membership_and_digital`** | 14 | 7.0% | 6.9% | Subscription fees & digital streaming. |
| **`order_cancellation_request`** | 13 | 6.5% | 6.5% | Pre-dispatch order cancellations. |
| **`account_access_and_security`** | 10 | 5.0% | 4.8% | Critical safety & account lockout edge cases. |
| **`other_unknown`** | 8 | 4.0% | 3.8% | General venting / ambiguous out-of-scope queries. |
| **Total** | **200** | **100.0%** | **100.0%** | **Balanced, representative evaluation corpus** |

---

### B. Actionable Triage / Handling Distribution
- **`AUTO_HANDLE`**: **140 examples** (**70.0%**)
- **`ESCALATE`**: **60 examples** (**30.0%**)
  - *Primary Escalation Reasons*:
    - Account Security / Password / OTP Lockout ($10$ cases, $16.7\%$ of escalations)
    - Severe Delivery Overdue ($>48$h past Prime date) ($14$ cases, $23.3\%$)
    - High-Risk Missing / Porch Stolen Delivery ($8$ cases, $13.3\%$)
    - Repeated Courier Pickup Failures ($8$ cases, $13.3\%$)
    - Double Charge / Disputed Billing ($7$ cases, $11.7\%$)
    - Severe Product Damage / Soap-in-box ($6$ cases, $10.0\%$)
    - Extended Refund Delays ($>14$ days) ($4$ cases, $6.7\%$)
    - Complex Supervisor Requests ($3$ cases, $5.0\%$)

---

## 5. Automated Isolation & Privacy Audit Results

Executed via `scripts/validate_golden_isolation.py`:
- **Conversation ID Overlap**: **0 IDs** (100% disjoint from dev samples).
- **Temporal Constraint**: **100% compliant** (all timestamps between `2017-11-27 13:01:51` and `2017-12-03 22:03:06`).
- **Exact Query Leakage**: **0 duplicate queries**.
- **PII Sanitization**: **100% scrubbed** (zero raw order numbers, emails, or phone numbers in inputs/labels).
