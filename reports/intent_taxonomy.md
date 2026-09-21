# Amazon Customer Support Intent Taxonomy Report

This document presents the finalized 11-intent taxonomy for AmazonHelp, derived empirically from real-world Twitter customer-support conversations and validated via a 100-case consistency audit.

---

## 1. Intent Discovery Methodology

Rather than imposing a generic conversational schema (such as Banking77), the intent taxonomy was derived directly from AmazonHelp data through a 4-stage empirical process:

1. **Corpus Sampling**: Extracted 8,000 representative conversation cases exclusively from the earlier 85% chronological development partition (`data/samples/intent_discovery_cases.parquet`).
2. **Linguistic & N-Gram Profiling**: Analyzed 1-to-3 gram frequencies, verb-noun collocations (*"track order"*, *"missed delivery"*, *"refund status"*, *"broken package"*).
3. **Unsupervised Topic Clustering**: Executed TF-IDF feature extraction with MiniBatch K-Means clustering ($K=12$) to surface semantic groupings.
4. **Historical Resolution Coupling**: Inspected paired agent responses for each topic cluster to ensure intent categories reflect distinct operational resolution workflows.

---

## 2. Taxonomy Summary & Empirical Frequencies

| Intent Name | Category Description | Approx. Frequency | Primary Resolution Action | Auto-Handle Suitability |
|---|---|---|---|---|
| **`delivery_status_tracking`** | In-transit tracking, carrier scans, dispatch queries | 14.5% | Provide self-serve 'Your Orders' tracking URL | **High** |
| **`late_delivery_complaint`** | Overdue delivery, missed Prime guaranteed date | 16.2% | Apologize, verify delivery buffer, check compensation | **Medium** |
| **`order_delivered_not_received`** | Marked delivered but package absent/missing | 7.8% | Property/neighbor search advice, 24h buffer, claim link | **Medium** |
| **`damaged_defective_or_wrong_item`**| Item broken, crushed box, wrong variant/item | 9.4% | Direct to Returns Center for free replacement/return | **High** |
| **`return_and_pickup_inquiry`** | Return procedure, label creation, courier pickup delay | 11.2% | Guide return process, reschedule pickup | **High** |
| **`refund_status_and_request`** | Pending refund timeline, missing credit, bank delay | 10.8% | Explain bank processing timeline (3-5 business days) | **High** |
| **`order_cancellation_request`** | Cancel active order, seller-cancelled inquiries | 6.5% | Guide pre-dispatch cancellation in 'Your Orders' | **High** |
| **`payment_and_billing_issues`** | Card decline, double charge, gift card balance | 7.1% | Check payment method, explain bank auth hold | **Medium** |
| **`prime_membership_and_digital`** | Prime fee, Prime Video subtitles, Echo/Kindle setup | 6.9% | Manage Prime portal link, digital troubleshooting | **High** |
| **`account_access_and_security`** | Login failure, OTP/password reset, locked/hacked account| 4.8% | Direct to Account Recovery portal | **Low (Escalate)** |
| **`other_unknown`** | General venting, unintelligible text, out-of-scope | 3.8% | Clarifying questions or empathetic acknowledgment | **Medium** |

---

## 3. Results of the 100-Case Consistency Audit

We performed a formal consistency audit on 100 randomly sampled customer cases ($N=100$, `seed=42`) from the development corpus:

- **Unambiguous Single-Intent Match**: **87.0%** (87 cases mapped directly to exactly one intent with high confidence).
- **Context-Dependent / Multi-Intent Ambiguity**: **9.0%** (9 cases required reading prior turns or applying the Primary Intent hierarchy).
- **`other_unknown` / Unintelligible**: **4.0%** (4 cases lacked actionable order/issue information).
- **Top Confusing Pairs Observed**:
  1. `late_delivery_complaint` vs. `delivery_status_tracking` (4 cases)
  2. `return_and_pickup_inquiry` vs. `refund_status_and_request` (3 cases)
  3. `payment_and_billing_issues` vs. `prime_membership_and_digital` (2 cases)

---

## 4. Multi-Intent Handling & Primary Intent Hierarchy

In approximately **11.4%** of customer queries, multiple complaints co-occur (e.g. *"My delivery was late and the box arrived damaged"*).

**Primary Intent Decision Rules**:
1. **Physical Defect Precedence**: Product damage or incorrect items take priority over delivery timing $\rightarrow$ `damaged_defective_or_wrong_item`.
2. **Financial Resolution Precedence**: Demanding money back after an order cancellation or return takes priority $\rightarrow$ `refund_status_and_request`.
3. **Security Precedence**: Account compromise or unauthorized charges take top priority for routing $\rightarrow$ `account_access_and_security`.

---

## 5. Artifact Links
- Machine-Readable Schema: [`data/processed/intent_taxonomy.json`](file:///e:/AI-Customer-Support-Agent/data/processed/intent_taxonomy.json)
- Human Annotation Guidelines: [`data/processed/intent_annotation_guidelines.md`](file:///e:/AI-Customer-Support-Agent/data/processed/intent_annotation_guidelines.md)
- Context Unit Analysis: [`reports/intent_context_analysis.md`](file:///e:/AI-Customer-Support-Agent/reports/intent_context_analysis.md)
- Boundary & Disambiguation Report: [`reports/intent_boundary_analysis.md`](file:///e:/AI-Customer-Support-Agent/reports/intent_boundary_analysis.md)
