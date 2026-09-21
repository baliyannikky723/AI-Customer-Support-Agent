# AI Customer Support Agent & Evaluation System (Hiver Take-Home Assignment)

An enterprise-grade, reproducible AI Customer Support Agent built on real-world multi-turn Twitter customer support conversations from the Kaggle **Customer Support on Twitter** dataset (`thoughtvector/customer-support-on-twitter`).

The system classifies customer inquiries into domain-specific intents, retrieves relevant historical resolution pairs, synthesizes policy-grounded replies, enforces post-generation safety guardrails, and deterministically triages tickets between **`AUTO_HANDLE`** and **`ESCALATE`** to prioritize customer safety.

---

## 🎯 Executive Overview & Project Goals

- **Target Brand**: **`AmazonHelp`** (Selected from 108 brands based on 155,445 multi-turn conversations and only 0.6% DM deflection).
- **Core Philosophy**: **"The proof is worth more than the system"** — rigorous experimental isolation, multi-baseline benchmarking, decoupled evaluation metrics, and zero data leakage.
- **Safety Priority**: Prioritizes avoiding **Dangerous False Auto-Handles** (preventing automated deflections on accounts requiring human verification) over inflating headline automation percentages.
- **Offline Reproducibility**: 100% of baseline evaluations, vector indexing, reranking, end-to-end agent traces, LLM-as-a-Judge scoring, and unit tests run offline in deterministic mock mode without requiring paid API keys.

---

## 🏗️ End-to-End Pipeline Architecture

```
                                [ Customer Message ]
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │   1. PII Sanitizer & Masking │
                         │ Masks [CUSTOMER], [ORDER_ID] │
                         └───────────────┬──────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │  2. 11-Intent Classifier     │
                         │ (TF-IDF + Logistic Regres.)  │
                         └───────────────┬──────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │ 3. Dense FAISS Vector Search │
                         │ (all-MiniLM-L6-v2 Embeddings)│
                         └───────────────┬──────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │ 4. Intent-Aware Reranker     │
                         │ (+0.10 Bonus, 0.60 Conf Gate)│
                         └───────────────┬──────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │ 5. Evidence Pack Selector    │
                         │ (Deduplication + Score [0,1])│
                         └───────────────┬──────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │ 6. LLM Response Generator    │
                         │ (Structured JSON Contract)   │
                         └───────────────┬──────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │ 7. Grounding Guardrails      │
                         │ (Blocks PII & Live Claims)   │
                         └───────────────┬──────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │ 8. Deterministic Triage      │
                         │ (Mandatory Escalation Rules) │
                         └───────────────┬──────────────┘
                                         │
                                         ▼
                       [ AUTO_HANDLE  vs.  ESCALATE ]
```

---

## 📊 Comprehensive Benchmark Results

Evaluated strictly on the **200 held-out golden evaluation examples** (`data/golden/golden_inputs.jsonl`, `data/golden/golden_labels.jsonl`):

| Evaluation Dimension / Metric | Baseline 1 (Majority Class) | Baseline 2 (TF-IDF + LR) | Final AI Agent (Phase 10) | Operational Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Intent Classification Accuracy** | `13.50%` | `70.50%` | **`70.50%`** | Consistent domain intent prediction |
| **Intent Macro F1 Score** | `0.0070` | `0.6915` | **`0.6915`** | Solid classification across 11 intents |
| **Retrieval Mean Top-1 Similarity** | `0.0000` | `0.2822` (TF-IDF) | **`0.7244`** (MiniLM) | **+156.7% relative gain over TF-IDF** |
| **Top-1 Same-Intent Match Rate** | `13.50%` | `~25.0%` | **`50.0%`** (100/200) | **+26.58% relative gain via reranking** |
| **Top-5 Same-Intent Coverage** | `N/A` | `~45.0%` | **`68.5%`** (137/200) | Rich multi-candidate policy presence |
| **Triage Routing Accuracy** | `70.00%` (Trivial) | `63.50%` | **`41.00%`** (Conservative) | Balanced, safety-first ticket routing |
| **Dangerous False AUTO_HANDLE** | `60 / 60` (100.0%) | `24 / 60` (40.0%) | **`13 / 60` (21.67%)** | **45.8% reduction vs TF-IDF, 78.3% vs Majority** |
| **Escalation Recall** | `0.00%` | `58.33%` | **`78.33%`** (47/60) | **Catches 78.3% of critical/sensitive cases** |
| **Escalation Precision** | `0.00%` | `43.21%` | **`30.92%`** (47/152) | Conservative human routing bias |
| **Escalation F1 Score** | `0.0000` | `0.4966` | **`0.4434`** | Robust safety-oriented routing |
| **AUTO_HANDLE Rate** | `100.0%` (200/200) | `59.5%` (119/200) | **`24.0%`** (48/200) | Only safely automates clean self-service cases |
| **ESCALATE Rate** | `0.0%` (0/200) | `40.5%` (81/200) | **`76.0%`** (152/200) | Explicit human escalation of high-risk queries |
| **PII / Live-Claim Leakage Rate** | `N/A` | `N/A` | **`0.0%`** (0/200) | 100% compliance with zero live claims |
| **LLM Judge Overall Quality (1–5)** | `N/A` | `N/A` | **`4.21 / 5.0`** | High-quality domain grounding (Mock Mode) |
| **Human Evaluation Agreement** | `N/A` | `N/A` | **`Pending Human Review`** | Ready-to-import template & calculator |
| **Unit Test Suite Pass Rate** | `N/A` | `28 / 28` | **`68 / 68 passing`** | 100% test pass rate in 14.5s |

---

## 🏷️ Brand Selection & Intent Taxonomy

### Brand Selection Evidence (from `reports/brand_candidates.md`)
Profiling 2,811,774 tweets across 108 brands confirmed **`AmazonHelp`** as the premier choice:
- **155,445 usable multi-turn conversations** and 169,840 support messages.
- **Lowest DM deflection rate (0.6%)** compared to AppleSupport (46.8%) and Uber_Support (35.6%), providing rich public resolution patterns.

### 11-Intent Domain Taxonomy
1. `delivery_status_tracking`
2. `late_delivery_complaint`
3. `order_delivered_not_received` *(Mandatory Escalation)*
4. `damaged_defective_or_wrong_item`
5. `return_and_pickup_inquiry`
6. `refund_status_and_request`
7. `order_cancellation_request`
8. `payment_and_billing_issues` *(Mandatory Escalation on Fraud/Double Charge)*
9. `prime_membership_and_digital`
10. `account_access_and_security` *(Mandatory Escalation)*
11. `other_unknown` *(Mandatory Escalation)*

---

## 🔒 Strict Golden Set Isolation & Leakage Audit

To prevent evaluation leakage:
- **Strict Chronological Cutoff**: The golden set contains 200 hand-validated interactions occurring strictly post-cutoff (`2017-11-27 13:01:51+00:00` to `2017-12-03 22:03:06+00:00`).
- **Zero ID Overlap**: Verified 0 conversation ID overlap across the 200 golden examples, 5,000 dev sample, and 8,000 retrieval index cases.
- **Zero Query Duplication**: Verified 0 exact text query string matches between evaluation inputs and training corpora.
- **Zero Tuning on Eval Sets**: Golden labels and human review templates were held out strictly for final evaluation.

---

## ⚖️ LLM-as-a-Judge Rubric & Human Annotation Framework

### LLM-as-a-Judge 5-Dimension Rubric (1–5 Likert Scale)
1. **Correctness (1–5)**: Does the reply appropriately address the specific customer problem?
2. **Groundedness (1–5)**: Are all policy claims supported by historical evidence without hallucinating commitments?
3. **Helpfulness (1–5)**: Does the reply provide clear, actionable self-service navigation paths?
4. **Safety (1–5)**: Does the reply avoid fake live database lookups, PII leaks, or unsafe automated actions?
5. **Tone (1–5)**: Is the response empathetic, professional, polite, and concise?

### Human Evaluation Package & Scientific Integrity
- **Status**: `HUMAN AGREEMENT NOT YET MEASURED`
- **Integrity Guarantee**: In strict accordance with assignment guidelines, **no fake human scores or synthetic Cohen's kappa values were fabricated**.
- **Deliverables**:
  - Stratified 50-Example Subset: [`data/evaluation/human_eval_set.jsonl`](file:///e:/AI-Customer-Support-Agent/data/evaluation/human_eval_set.jsonl)
  - Human Annotation Template: [`data/evaluation/human_annotations_template.jsonl`](file:///e:/AI-Customer-Support-Agent/data/evaluation/human_annotations_template.jsonl)
  - Reviewer Guidelines: [`reports/human/annotation_guidelines.md`](file:///e:/AI-Customer-Support-Agent/reports/human/annotation_guidelines.md)
  - Validation & Import Tool: [`scripts/import_human_annotations.py`](file:///e:/AI-Customer-Support-Agent/scripts/import_human_annotations.py)
  - Agreement Calculator: [`scripts/analyze_judge_human_agreement.py`](file:///e:/AI-Customer-Support-Agent/scripts/analyze_judge_human_agreement.py)

---

## 🔍 Top 5 Failure Modes Analysis

1. **Multi-Turn Delay Frustration (13 cases, 6.5%)**: Repeated carrier delays misclassified under standard tracking due to lexical similarity.
   - *Mitigation*: Add customer sentiment intensity and conversation turn-depth thresholds to `TriageEngine`.
2. **Intent Boundary Ambiguity (59 cases, 29.5%)**: Boundary cases between account security and prime digital resulting in lower classifier confidence (0.34).
   - *Mitigation*: Train a multi-label classification head or fine-tuned RoBERTa cross-encoder.
3. **Third-Party Marketplace Discrepancies (4 cases, 2.0%)**: Defective items sold by 3P merchants treated under standard Amazon fulfillment policies.
   - *Mitigation*: Add dedicated 3P marketplace seller dispute taxonomy intent.
4. **Hardware / Digital Overlap (6 cases, 3.0%)**: Social praise for hardware (*"love my Echo"*) misclassified under Prime digital membership.
   - *Mitigation*: Add dedicated `social_chitchat_feedback` intent bucket.
5. **Doorstep Cash Load Refund Delays (3 cases, 1.5%)**: Offline cash-on-delivery refund turnaround differs from online card refunds.
   - *Mitigation*: Expand retrieval vector index with localized cashload FAQ resolutions.

---

## ⚠️ "What Is Misleading About My Headline Number?"

1. **Headline Triage Accuracy is Counter-Intuitive**: The trivial Majority baseline achieved a 70% triage accuracy by auto-handling 100% of tickets, yet failed on 100% of dangerous security and billing cases (60/60). Our agent achieves 41.0% accuracy because it prioritizes safety, achieving **78.33% escalation recall** and cutting dangerous false auto-handles to 13.
2. **Mock Judge Determinism vs. Live Stochasticity**: The 4.21/5 overall judge score reflects template-grounded deterministic mock responses. A live LLM introduces non-zero variance that requires continuous guardrail validation.
3. **High Semantic Similarity Does Not Equal Correctness**: A retrieved case can have 0.85 cosine similarity but represent an opposing policy context (e.g. digital streaming vs physical returns).

---

## 🚫 "What We Did NOT Build": Explicit Scope Boundaries

This prototype strictly does **NOT**:
- Connect to live Amazon internal databases or carrier scan APIs.
- Fabricate real-time order tracking status or claim live account lookup capabilities.
- Execute automated financial transactions (initiating irreversible refunds or bank transfers).
- Perform autonomous account authentication changes (password resets or 2FA modifications).

---

## 🚀 15-Minute Offline Reproduction Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/baliyannikky723/AI-Customer-Support-Agent.git
cd AI-Customer-Support-Agent

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

### 2. Step-by-Step Reproduction Commands (100% Offline & Deterministic)
```bash
# 1. Evaluate Baseline 1 (Majority) & Baseline 2 (TF-IDF + LR)
python scripts/evaluate_baselines.py

# 2. Build MiniLM FAISS Vector Index (8,000 cases)
python scripts/build_semantic_index.py

# 3. Evaluate Dense Semantic Retrieval (Phase 6)
python scripts/evaluate_semantic_retrieval.py

# 4. Evaluate Intent-Aware Soft Candidate Reranking (Phase 7)
python scripts/evaluate_intent_aware_retrieval.py

# 5. Evaluate Full End-to-End AI Support Agent with Safety Triage (Phase 8)
python scripts/evaluate_phase8_agent.py

# 6. Run LLM-as-a-Judge Evaluation (Phase 9)
python scripts/run_llm_judge.py

# 7. Generate 50-Example Stratified Human Evaluation Set
python scripts/build_human_eval_set.py

# 8. Validate Human Annotation Import Status
python scripts/import_human_annotations.py

# 9. Compute Judge vs. Human Agreement (or generate guidelines & status)
python scripts/analyze_judge_human_agreement.py

# 10. Run Full Unit Test Suite (68 passing tests)
python -m unittest discover tests
```

---

## 📁 Repository Structure & Artifact Map

```
AI-Customer-Support-Agent/
├── configs/
│   └── default_config.yaml                  # System configuration (retrieval, guardrails, triage, LLM)
├── data/
│   ├── evaluation/
│   │   ├── human_eval_set.jsonl             # 50-example stratified human evaluation benchmark
│   │   └── human_annotations_template.jsonl # Reviewer template with 1–5 rubric scoring fields
│   ├── golden/
│   │   ├── golden_inputs.jsonl              # 200 held-out golden customer queries (post-cutoff)
│   │   └── golden_labels.jsonl              # Ground-truth intent & triage routing labels
│   ├── processed/
│   │   ├── faiss_index.bin                  # Dense MiniLM FAISS vector index (8,000 cases)
│   │   └── resolution_metadata.parquet      # Historical metadata pairs & intent labels
│   └── samples/
│       ├── amazonhelp_dev_sample.parquet    # 5,000 pre-cutoff development conversations
│       └── intent_discovery_cases.parquet   # 8,000 pre-cutoff historical training cases
├── reports/
│   ├── final_report.md                      # Comprehensive 6-Page Technical Report
│   ├── submission_checklist.md              # 18-point submission verification checklist
│   ├── final_artifact_inventory.md          # Exhaustive artifact index and purpose mapping
│   ├── human_evaluation_rubric.md           # Human reviewer grading rubric & guidelines
│   ├── baseline_results/                    # Baseline 1 (Majority) & Baseline 2 (TF-IDF) reports
│   ├── semantic_retrieval_results/          # Phase 6 MiniLM dense retrieval metrics
│   ├── intent_aware_retrieval_results/      # Phase 7 Intent-aware reranking metrics
│   ├── phase8_results/                      # Phase 8 End-to-end agent triage & prediction traces
│   ├── judge/                               # LLM-as-a-Judge rubric, metrics, and summary
│   └── human/                               # Human annotation guidelines & agreement status
├── scripts/
│   ├── evaluate_baselines.py                # Evaluates Majority & TF-IDF baselines on 200 golden cases
│   ├── build_semantic_index.py              # Builds MiniLM dense FAISS vector index
│   ├── evaluate_semantic_retrieval.py       # Evaluates pure dense vector retrieval (Top-k=5)
│   ├── evaluate_intent_aware_retrieval.py   # Evaluates soft intent reranking (+0.10 bonus)
│   ├── evaluate_phase8_agent.py             # Evaluates complete AI Support Agent on 200 golden cases
│   ├── run_llm_judge.py                     # Runs multi-dimensional LLM Judge evaluation
│   ├── build_human_eval_set.py              # Stratified sampler generating 50-case human subset
│   ├── import_human_annotations.py          # Validates & imports human reviewer annotations
│   └── analyze_judge_human_agreement.py     # Calculates weighted Cohen's kappa & MAD agreement
├── src/                                     # Modular source code (preprocessing, intents, retrieval, generation, escalation, judge)
├── tests/                                   # Full unit test suite (68 passing tests)
├── decision_log.md                          # 37 chronological non-obvious engineering decisions
├── README.md                                # Project documentation & reproduction guide
└── .env.example                             # Safe environment variable configuration template
```

---

## 🗓️ "One-More-Week" Engineering Roadmap

1. **Day 1–2 (Multi-Turn Turn Depth Modeling)**: Integrate conversation turn count and negative customer sentiment intensity into the triage engine to eliminate the 13 missed escalations.
2. **Day 3 (Cross-Encoder Intent Boundary Hardening)**: Train a fine-tuned RoBERTa cross-encoder on intent boundary edge cases to improve intent accuracy beyond 70.5%.
3. **Day 4 (Live LLM Evaluation & Rubric Calibration)**: Run real Gemini/GPT-4o judges against the 50-example human review annotations to measure empirical Cohen's kappa.
4. **Day 5 (3P Marketplace Specialization)**: Expand retrieval corpus with dedicated 3P seller dispute guidelines and cash-on-delivery turnaround protocols.
5. **Day 6–7 (Authenticated Tool Integration Scaffolding)**: Implement OAuth-gated tool abstraction for live order inspection without exposing customer PII.

---

## 📄 License & Attribution

- **Dataset**: [Customer Support on Twitter (Kaggle)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (`thoughtvector/customer-support-on-twitter`).
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (Apache 2.0 License).
- **Target Enterprise**: `AmazonHelp` (Analyzed for academic/demonstration purposes under fair use).