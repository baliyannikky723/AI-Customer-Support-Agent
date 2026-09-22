# AI Customer Support Agent & Evaluation System: Final Technical Report
**Target Brand**: `AmazonHelp` | **Corpus**: Kaggle Twitter Customer Support Dataset (~3M tweets) | **Version**: 1.0 (Phase 10 Submission)

---

## 1. Problem Framing, Success Criteria & Scope Boundaries

### 1.1 Problem Statement
Customer support operations at high-volume e-commerce enterprises face severe trade-offs between automated self-service efficiency and customer safety. Automated agents must accurately classify multi-turn customer intents, retrieve relevant historical resolutions, and draft grounded, policy-compliant replies. Crucially, when faced with high-risk scenarios (account lockouts, fraud, carrier loss, low confidence), the system must reliably **escalate to human specialists** rather than hallucinating live status or fabricating unauthorized commitments.

### 1.2 "What Good Means": Rigorous Multi-Dimensional Success Criteria
In this project, system performance is evaluated across four decoupled, measurable pillars:
1. **Classification Integrity**: High intent macro-F1 across 11 domain intents without collapsing into majority-class predictions.
2. **Retrieval Groundedness**: High semantic alignment ($\text{sim} \ge 0.70$) between customer queries and verified historical resolution pairs.
3. **Safety-Critical Triage**: Minimizing **Dangerous False Auto-Handles** (preventing automated deflections on accounts requiring human verification) while maintaining high **Escalation Recall** ($\ge 75\%$).
4. **Natural Language Grounding**: Zero fabricated live-system claims (*"I checked your order"*), zero unmasked PII leakage, and zero stale historical timestamp leakage.

### 1.3 "What We Did NOT Build": Explicit Scope & Boundary Declaration
To maintain strict empirical rigor, this prototype explicitly does **NOT**:
- Connect to live Amazon internal databases or carrier scan APIs.
- Fabricate real-time order tracking status or claim live account lookup capabilities.
- Execute automated financial transactions (initiating irreversible refunds or bank transfers).
- Perform autonomous account authentication changes (password resets or 2FA modifications).

---

## 2. Dataset Exploration, Brand Selection & Intent Taxonomy

### 2.1 Dataset Profiling & Brand Selection Evidence
Profiling all 2,811,774 tweets across 108 brands identified **`AmazonHelp`** as the optimal target enterprise:
- **Volume & Substantive Responses**: 155,445 usable multi-turn conversations and 169,840 support messages.
- **Low Canned Deflection**: Only **0.6%** direct-message (DM) deflections (compared to 46.8% for `@AppleSupport` and 35.6% for `@Uber_Support`), providing rich public resolution patterns.
- **Domain Diversity**: Distinct operational policies across delivery tracking, returns, damaged items, Prime digital subscriptions, and billing disputes.

### 2.2 11-Intent Domain Taxonomy
Derived empirically from iterative thematic coding over 8,000 AmazonHelp conversation threads:
1. `delivery_status_tracking`
2. `late_delivery_complaint`
3. `order_delivered_not_received` *(High Risk)*
4. `damaged_defective_or_wrong_item`
5. `return_and_pickup_inquiry`
6. `refund_status_and_request`
7. `order_cancellation_request`
8. `payment_and_billing_issues` *(High Risk)*
9. `prime_membership_and_digital`
10. `account_access_and_security` *(High Risk)*
11. `other_unknown`

### 2.3 Strict Golden Set Isolation & Privacy Audit
The final evaluation benchmark comprises **200 hand-validated, stratified golden examples** (`data/golden/golden_inputs.jsonl`, `data/golden/golden_labels.jsonl`):
- **Temporal Cutoff**: Golden queries span strictly post-cutoff timestamps (`2017-11-27 13:01:51+00:00` to `2017-12-03 22:03:06+00:00`), while training/retrieval corpora are restricted to pre-cutoff history.
- **Zero Leakage**: 0 conversation ID overlap (verified across 5,000 dev sample and 8,000 retrieval index cases) and 0 exact query duplicates.
- **PII Scrubbing**: 100% regex sanitization masking handles (`[CUSTOMER]`), order IDs (`[ORDER_ID]`), emails (`[EMAIL]`), and phone numbers (`[PHONE]`).

---

## 3. End-to-End System Architecture

```
                                Customer Inquiry
                                       │
                                       ▼
                       [ 1. PII Sanitization & Masking ]
                                       │
                                       ▼
                      [ 2. 11-Intent TF-IDF Classifier ]
                      (Predicts Intent + Probability)
                                       │
                                       ▼
                    [ 3. Dense MiniLM FAISS Vector Index ]
                     (Retrieves Top-10 Candidate Pairs)
                                       │
                                       ▼
                   [ 4. Intent-Aware Candidate Reranker ]
                    (Soft Intent Compatibility Bonus +0.10)
                                       │
                                       ▼
                    [ 5. Evidence Selection & Quality ]
               (Deduplication + Quality Score Heuristic [0, 1])
                                       │
                                       ▼
                  [ 6. LLM Response Generation (Mock/Real) ]
                   (Structured JSON Contract: Reply + Claims)
                                       │
                                       ▼
                  [ 7. Deterministic Grounding Guardrail ]
               (Blocks PII, Live Claims & 2017 Timestamps)
                                       │
                                       ▼
                    [ 8. Deterministic Safety Triage ]
               (Mandatory Escalation vs Grounded Auto-Handle)
                                       │
                                       ▼
                 [ 9. Final Decision: AUTO_HANDLE | ESCALATE ]
```

---

## 4. Baselines vs. Semantic Retrieval & Intent-Aware Reranking

### 4.1 Quantitative Progression across Pipeline Phases

| Component / Metric | Baseline 1 (Majority Class) | Baseline 2 (TF-IDF + LR) | Phase 6 (Dense Semantic) | Phase 7 (Intent-Aware) | Final Agent (Phase 8/9) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Intent Macro F1** | `0.0070` | `0.6915` | `0.6915` | `0.6915` | **`0.6915`** |
| **Intent Accuracy** | `13.50%` | `70.50%` | `70.50%` | `70.50%` | **`70.50%`** |
| **Retrieval Mean Top-1 Sim** | `N/A` | `0.2822` (TF-IDF) | `0.7280` (MiniLM) | **`0.7244`** | **`0.7244`** |
| **Top-1 Same-Intent Match** | `13.50%` | `~25.0%` | `39.5%` (79/200) | **`50.0%`** (100/200)| **`50.0%`** (100/200) |
| **Top-5 Same-Intent Coverage**| `N/A` | `~45.0%` | `65.5%` (131/200) | **`68.5%`** (137/200)| **`68.5%`** (137/200) |
| **Triage Accuracy** | `70.00%` (Trivial) | `63.50%` | — | — | **`41.00%`** (Conservative) |
| **Dangerous False Auto** | `60 / 60` (100%) | `24 / 60` (40.0%) | — | — | **`13 / 60` (21.67%)** |
| **Escalation Recall** | `0.00%` | `58.33%` | — | — | **`78.33%`** (47/60) |
| **Guardrail Pass Rate** | `N/A` | `N/A` | — | — | **`100.0%`** (200/200) |

### 4.2 Key Engineering Insights on Retrieval
- **TF-IDF Retrieval Failure**: Lexical TF-IDF retrieval averaged only 0.2822 cosine similarity, with only 5/200 queries exceeding 0.50 similarity.
- **Dense Vector Retrieval Leap**: MiniLM FAISS retrieval boosted mean similarity to **0.7280**, achieving $\ge 0.50$ similarity on 98.0% of test queries.
- **Intent-Aware Reranking**: Adding a soft compatibility bonus ($\beta = +0.10$) gated by a confidence threshold ($\tau = 0.60$) increased top-1 intent match from **39.5% to 50.0%** (+26.58% relative gain) while preserving dense semantic proximity.

---

## 5. Response Generation, Safety Guardrails & Triage Engine

### 5.1 Historical Responses as Evidence, Not Absolute Truth
Historical support replies reflect past interactions and contain case-specific artifacts. The prompt explicitly instructs the generator: *"Historical cases are examples of past resolution patterns, NOT proof that the same condition or action applies to the current user."*

### 5.2 Deterministic Grounding Guardrail
Post-generation regex guardrails inspect all generated responses:
1. **PII Leakage**: Rejects unmasked personal handles, emails, phone numbers, and raw order numbers.
2. **Forbidden Live-System Verbs**: Detects and rejects hallucinated backend actions (*"I checked your account"*, *"Your package is currently in transit at hub"*).
3. **Historical Timestamps**: Blocks stale 2017 dates from Kaggle evidence.
4. **Validation Performance**: **100.0% pass rate** across all 200 golden responses.

### 5.3 Deterministic Triage Engine & Mandatory Escalation Rules
To prevent catastrophic automated failures, the triage engine enforces non-negotiable escalation rules:
- **`ACCOUNT_SECURITY_ACTION`**: Mandates human escalation for account lockouts, 2FA, and compromised credentials.
- **`MISSING_PACKAGE_CLAIM`**: Escalates orders marked delivered but missing on porch for carrier claims.
- **`PAYMENT_DISPUTE_VERIFICATION`**: Escalates unauthorized charges and duplicate billing disputes.
- **`LOW_INTENT_CONFIDENCE`**: Escalates queries where classifier confidence $< 0.60$.
- **`INSUFFICIENT_EVIDENCE_QUALITY`**: Escalates when heuristic evidence quality score $< 0.50$.

---

## 6. Comprehensive Evaluation, Failure Analysis & Roadmap

### 6.1 LLM-as-a-Judge Rubric Evaluation (200 Golden Cases)
Evaluated across 5 standardized dimensions (1–5 Likert scale) using `MockLLMJudge`:
- **Correctness**: `3.58 / 5.0` (Core inquiry correctly diagnosed and addressed)
- **Groundedness**: `4.68 / 5.0` (Strict evidence grounding with zero hallucinated commitments)
- **Helpfulness**: `4.57 / 5.0` (Direct self-service navigation in 'Your Orders')
- **Safety**: `4.05 / 5.0` (Conservative handling with zero live claim breaches)
- **Tone**: `4.58 / 5.0` (Empathetic, courteous, professional)
- **Overall Quality**: `4.21 / 5.0` | **Acceptance Rate**: `100.0%` | **Critical Failure Rate**: `0.0%`

### 6.2 Human Annotation Framework & Agreement Status
- **Status**: `HUMAN AGREEMENT NOT YET MEASURED`
- **Integrity Statement**: In strict compliance with assignment guidelines, **no fake human scores or synthetic Cohen's kappa values were fabricated**.
- **Deliverables Ready**: 50-example stratified evaluation set (`data/evaluation/human_eval_set.jsonl`), review template (`data/evaluation/human_annotations_template.jsonl`), import tool (`scripts/import_human_annotations.py`), and agreement calculator (`scripts/analyze_judge_human_agreement.py`).

### 6.3 Top 5 Response-Quality Failure Modes

| # | Failure Mode | Freq (%) | Real Example Query | Root Cause | Automated Detection | Proposed Mitigation |
|---|---|---|---|---|---|---|
| **1** | **Multi-Turn Delay Frustration** | 13 (6.5%) | `Amazon prime pushed back my order... Now it’s 2 days late false advertisement 😳😠` | High lexical overlap to self-service tracking despite repeated delay escalation cues. | Flagged as dangerous false auto in triage audit. | Add negative sentiment intensity & turn depth gates. |
| **2** | **Intent Boundary Ambiguity** | 59 (29.5%) | `14 captcha codes in a row - have you been hacked?` | Low classifier confidence (0.34) across account security vs prime digital. | Correctly escalated via `LOW_INTENT_CONFIDENCE`. | Fine-tune cross-encoder or multi-label head. |
| **3** | **3P Marketplace Discrepancies** | 4 (2.0%) | `seller sent defective product and wants ME to cancel on my end...` | Complex 3P marketplace seller policies treated as standard Amazon fulfillment. | Correctly escalated via `LOW_INTENT_CONFIDENCE`. | Add dedicated 3P marketplace seller dispute taxonomy intent. |
| **4** | **Hardware / Digital Overlap** | 6 (3.0%) | `In love with amazon echo #alexa #echo 😀 — listening to Music` | Ambiguous praise misclassified under `prime_membership_and_digital`. | Scored 3/5 correctness for generic prime management reply. | Add dedicated `social_feedback` intent bucket. |
| **5** | **Doorstep Cashload Delay** | 3 (1.5%) | `pls return my money (Doorstep Cashload) u r taking 2 much time.` | Offline cash load refund timelines differ from online credit card refund timelines. | Correctly escalated to human support. | Expand evidence base with localized COD/cashload FAQs. |

### 6.4 "What Is Misleading About My Headline Number?"
1. **Headline Triage Accuracy is Counter-Intuitive**: The Majority baseline achieved 70% triage accuracy by auto-handling 100% of tickets, yet caused catastrophic failure on 100% of dangerous cases (60/60). Our agent achieves 41.0% accuracy because it aggressively escalates low-confidence queries, achieving **78.33% escalation recall** and cutting dangerous false auto-handles to 13.
2. **Mock Judge Determinism vs. Live Stochasticity**: The 4.21/5 overall judge score reflects template-grounded deterministic mock responses. A live LLM introduces non-zero variance that requires continuous guardrail validation.
3. **High Semantic Similarity Does Not Equal Correctness**: A retrieved case can have 0.85 cosine similarity but represent an opposing policy context (e.g. digital streaming vs physical returns).

### 6.5 Realistic "One-More-Week" Engineering Plan
1. **Day 1–2 (Contextual Multi-Turn Modeling)**: Incorporate turn history and customer sentiment intensity into the triage engine to eliminate the remaining 13 missed escalations.
2. **Day 3 (Intent Boundary Hardening)**: Train a RoBERTa cross-encoder head on intent boundary edge cases to improve intent accuracy beyond 70.5%.
3. **Day 4 (Live LLM Evaluation & Rubric Calibration)**: Run real Gemini/GPT-4o judges against the 50-example human review annotations to measure empirical Cohen's kappa.
4. **Day 5 (3P Marketplace Specialization)**: Expand retrieval corpus with dedicated 3P seller dispute guidelines and cash-on-delivery turnaround protocols.
5. **Day 6–7 (Authenticated Tool Integration Scaffolding)**: Implement OAuth-gated tool abstraction for live order inspection without exposing customer PII.

---

## 7. Submission Verification & Final Results Summary Table

| Evaluation Dimension | Trivial / Baseline 1 | Simple / Baseline 2 | Final AI Agent (Phase 10) | Evidence & Verification Artifact |
| :--- | :--- | :--- | :--- | :--- |
| **Intent Classification** | Macro F1: `0.0070` | Macro F1: `0.6915` | **Macro F1: `0.6915` (Acc: `70.5%`)** | `reports/intent_classification_report.md` |
| **Retrieval Quality** | Cosine Sim: `0.00` | Cosine Sim: `0.2822` | **Cosine Sim: `0.7244` (Top-1 Match: `50.0%`)**| `reports/intent_aware_retrieval_results/` |
| **Dangerous False Auto** | `60 / 60` (100%) | `24 / 60` (40.0%) | **`13 / 60` (21.67%)** | `reports/phase8_results/metrics.json` |
| **Escalation Recall** | `0.00%` | `58.33%` | **`78.33%`** (47/60) | `reports/phase8_results/metrics.json` |
| **Guardrail Compliance** | `N/A` | `N/A` | **`100.0%` (0 PII / 0 Live Claims)** | `src/generation/grounding_guardrail.py` |
| **LLM Judge Score** | `N/A` | `N/A` | **`4.21 / 5.0` (Acceptable: `100.0%`)** | `reports/judge/judge_metrics.json` |
| **Human Agreement** | `N/A` | `N/A` | **`Pending Human Annotation`** | `reports/human/agreement_metrics.json` |
| **Unit Test Suite** | `N/A` | `28 passing` | **`68 / 68 passing` (13.0s runtime)** | `tests/` |
