# Engineering Decision Log

This document records the critical, non-obvious architectural, algorithmic, and data decisions made throughout the lifecycle of this project, along with the trade-offs and rationales.

---

| # | Decision Topic | Decision Made | Alternatives Considered | Rationale & Trade-offs | Phase |
|---|----------------|---------------|-------------------------|------------------------|-------|
| 1 | **Brand Selection** | Selected **`AmazonHelp`** as the single target brand. | `AppleSupport`, `Uber_Support`, `SpotifyCares`, `Delta` | `AmazonHelp` had the highest volume of substantive, public in-tweet solutions (only 0.6% DM deflection vs >35% for Apple/Uber) and clear operational intent boundaries (Shipping, Returns, Refunds, Prime, Damaged Items). | Phase 1 |
| 2 | **Language Scope & Filtering** | Adopted **`english_only`** as default mode via configurable parameter (`configs/default_config.yaml`). | Multilingual joint embedding, blindly dropping foreign rows without detection. | English accounts for ~79% of the corpus (~290k messages). Isolating English ensures semantic stability for intent classification and high LLM judge alignment without multilingual embedding drift. | Phase 2 |
| 3 | **PII Sanitization Architecture** | Built deterministic regex scrubber (`PIISanitizer`) masking Order IDs (`[ORDER_ID]`), emails (`[EMAIL]`), phones (`[PHONE]`), URLs (`[URL]`), and customer handles (`[CUSTOMER]`). | LLM-based NER (too slow/costly for 367k rows), leaving raw PII intact. | Regex runs in milliseconds across 367k rows with zero API cost, eliminating privacy risks and preventing LLMs from hallucinating or memorizing actual customer order IDs. | Phase 2 |
| 4 | **Resolution Deduplication Strategy** | Identified that 10.13% of support replies share identical canned templates. Retained frequency metadata while deduplicating representative templates for retrieval. | Aggressive deletion of duplicate text, naive indexing of all 169k raw support messages. | Naive indexing causes vector search collapse into generic deflection templates (*"Please contact us at [URL]"*). Deduplication preserves solution variety while tracking empirical policy frequency. | Phase 2 |
| 5 | **Leakage Prevention & Split Strategy** | Implemented **Chronological (Time-Aware) 85% Train/Dev vs 15% Eval Split** with hard disjoint conversation ID isolation. | Random 85/15 row split, random conversation split. | Random splitting causes temporal lookahead and near-neighbor leakage. Chronological splitting strictly mimics real-world production deployment where the agent is evaluated only on future unseen inquiries. | Phase 2 |
| 6 | **Domain-Grounded Intent Discovery** | Derived taxonomy directly from AmazonHelp data (10 operational retail intents + 1 `other_unknown`). | Adopting Banking77 (77 bank-specific intents) or generic dialogue acts. | Banking77 is irrelevant to retail logistics. Grounding taxonomy in real Amazon resolutions ensures tight coupling between classified intent, RAG retrieval policies, and auto-handling rules. | Phase 3 |
| 7 | **Delivery Intent Disambiguation** | Disentangled `delivery_status_tracking` (in-transit status) from `late_delivery_complaint` (overdue/missed Prime date) and `order_delivered_not_received` (marked delivered but missing). | Single monolithic "delivery_issue" category. | These 3 states trigger fundamentally different business actions: self-serve tracking link vs. delay compensation/carrier probe vs. stolen package claim. | Phase 3 |
| 8 | **Multi-Intent Primary Hierarchy** | Established a clear 3-tier precedence hierarchy (Security > Product Defect > Financial Action > Delivery Status). | Complex multi-label classification pipeline. | 11.4% of queries contain multiple complaints. A deterministic primary hierarchy enables sharp single-label classifier evaluation and unambiguous human annotation. | Phase 3 |
| 9 | **Input vs. Label Decoupling in Golden Set** | Maintained physically separated `golden_inputs.jsonl` and `golden_labels.jsonl` files linked via `example_id`. | Single unified evaluation JSON containing labels and reference answers in prompt input. | Physical file separation guarantees that the model runner cannot accidentally access ground-truth labels during inference. | Phase 4 |
| 10 | **Formal Handling & Escalation Ground-Truth Policy** | Defined explicit, objective rules for `AUTO_HANDLE` vs `ESCALATE` in `data/golden/handling_policy.md` (70% Auto / 30% Escalate). | Arbitrary ad-hoc escalation labeling. | Grounded triage policy establishes realistic operational thresholds (e.g. mandatory escalation for account lockouts, severe delays >48h, and suspected theft). | Phase 4 |
| 11 | **Two-Tier Non-LLM Baselines** | Implemented Baseline 1 (Majority Class Heuristic) and Baseline 2 (TF-IDF + Logistic Regression + TF-IDF Retriever). | Evaluating LLM without non-LLM baselines or using fine-tuned transformer as baseline. | Establishes transparent empirical lower and middle bounds to measure the genuine value-add of LLM semantic reasoning and RAG. | Phase 5 |
| 12 | **Macro F1 as Primary Performance Metric** | Emphasized Macro F1 over raw accuracy across all 11 intents. | Raw accuracy or micro F1. | Raw accuracy is heavily distorted by frequent intents (`late_delivery_complaint` is 16% vs `account_access_and_security` at 5%). Macro F1 treats safety-critical minority intents equally. | Phase 5 |
| 13 | **Safety-Critical Triage Error Tracking** | Explicitly tracked and counted **Dangerous False `AUTO_HANDLE`** decisions (predicting Auto when Ground Truth is Escalate). | Reporting only overall binary accuracy. | In customer support, failing to escalate an account compromise or stolen delivery creates catastrophic trust failure; accuracy hides high-consequence triage errors. | Phase 5 |

---

## Detailed Decision Notes

### Decision 1: Brand Selection — AmazonHelp
- **Context & Problem Statement**: The Kaggle dataset contains 108 active brands. We needed to choose one brand that affords rigorous intent classification, high-quality RAG grounding, and meaningful escalation triage.
- **Decision Taken**: Chose `AmazonHelp`.
- **Alternatives Considered**: `AppleSupport` (106k convs), `Uber_Support` (55k convs), `SpotifyCares` (41k convs), `Delta` (36k convs).
- **Rationale & Evidence**:
  - `AmazonHelp` contains **155,445 usable conversation pairs**.
  - Crucially, only **0.6%** of initial AmazonHelp responses are immediate DM deflections, compared to **46.8% for Apple** and **35.6% for Uber**. Amazon agents provide concrete policy advice in public tweets.
  - Distinct operational intents: Delivery Tracking, Return Procedure, Refund Request, Damaged/Missing Item, Prime Subscription, Payment/Billing.
- **Trade-offs**: ~22% multilingual presence handled via language preprocessing.

---

### Decision 2: Language Filtering & Preprocessing
- **Context & Problem Statement**: Amazon operates globally, leading to Spanish, Japanese, German, and French customer queries.
- **Decision Taken**: Implemented a lightweight, multi-script `LanguageDetector` and configured `language.mode: "english_only"` by default.
- **Rationale & Evidence**: 289,902 English messages provide more than sufficient scale (~65,265 complete English conversations) while preventing semantic cross-lingual noise in intent discovery and LLM judge scoring.

---

### Decision 3: Regex-Based PII Scrubbing
- **Context & Problem Statement**: Customer queries contain sensitive unmasked 17-digit Amazon order numbers (`112-3456789-1234567`), email addresses, and phone numbers.
- **Decision Taken**: Developed `PIISanitizer` with standard token placeholders (`[ORDER_ID]`, `[EMAIL]`, `[PHONE]`, `[URL]`, `[CUSTOMER]`).
- **Rationale & Evidence**: Deterministic masking protects privacy and ensures vector embeddings cluster by semantic customer intent rather than arbitrary order ID tokens.

---

### Decision 4: Response Pattern & Canned Template Management
- **Context & Problem Statement**: Support tweets prepend unique customer handles (`@105834`) and append unique employee signatures (`^SN`), making 99.98% of raw tweets technically unique, despite using repeated template phrasing.
- **Decision Taken**: Normalized customer handles and signatures during RAG preprocessing to expose the 10.13% underlying template repetitions, ensuring diverse retrieval.

---

### Decision 5: Chronological Partitioning to Eliminate Retrieval Leakage
- **Context & Problem Statement**: RAG evaluation is vulnerable to self-retrieval leakage if the test set is sampled randomly from the vector corpus.
- **Decision Taken**: Enforced a strict chronological split (85% Train/Dev vs 15% Evaluation Set) with disjoint conversation ID validation.

---

### Decision 14: Embedding Model Selection (`sentence-transformers/all-MiniLM-L6-v2`)
- **Context & Problem Statement**: Dense vector retrieval requires a performant, lightweight, reproducible embedding model that operates with low CPU latency and minimal RAM footprint.
- **Decision Taken**: Selected `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, 22.7M parameters) with L2 normalization.
- **Rationale & Evidence**:
  - Encodes 8,000 cases in 70.5s on CPU (113.4 cases/sec) with an index binary size of only 11.72 MB.
  - Generates compact 384-dimensional embeddings that preserve semantic intent without the 2x-4x memory overhead of 768-dim models.
  - Achieved a **+157.97% relative improvement** in mean top-1 cosine similarity over the TF-IDF baseline (0.7280 vs 0.2822).
- **Trade-offs**: Slightly shorter maximum sequence length (256 tokens), which is well above the 28.4 token average for Twitter support messages.

---

### Decision 15: FAISS Vector Index Architecture (`IndexFlatIP`)
- **Context & Problem Statement**: Need to choose between Approximate Nearest Neighbor (ANN) index structures (IVF, HNSW) vs Exact Search (`IndexFlatIP`).
- **Decision Taken**: Adopted `faiss.IndexFlatIP` paired with L2-normalized vector embeddings.
- **Rationale & Evidence**:
  - Normalizing embeddings converts Inner Product search into exact Cosine Similarity.
  - For $N=8,000$ cases, exact brute-force search executes in under 2.5 ms per query on CPU.
  - Avoids ANN quantization error, clustering centroid variability, and hyperparameter tuning overhead.

---

### Decision 16: Corpus Construction and Exact Deduplication Strategy
- **Context & Problem Statement**: Repetitive support interactions can cause vector retrieval to collapse into generic deflection links or duplicate entries.
- **Decision Taken**: Deduplicated exact `(customer_query, amazon_response)` pairs while preserving natural distribution of resolution policies across distinct queries.
- **Rationale & Evidence**:
  - 97.1% of historical responses in the 8,000 case dev pool are unique resolution variations.
  - Retaining diverse historical replies gives the downstream LLM rich context on how human agents navigated nuanced customer scenarios.

---

### Decision 17: Query Representation Strategy (`current_only` as Default)
- **Context & Problem Statement**: Customer inquiries in Twitter threads often have multi-turn history. Should the retriever embed the current message only or concatenate all preceding turns?
- **Decision Taken**: Defaulted to `current_only` with configurable support for `context_plus_current`.
- **Rationale & Evidence**:
  - Customer opening messages contain dense issue specifics (tracking numbers, damage descriptions, refund delays).
  - Blindly appending greeting filler or bot prompts adds semantic noise that degrades nearest-neighbor alignment.

---

### Decision 18: Separation of Intent Filtering from Vector Retrieval
- **Context & Problem Statement**: Golden intent labels exist for the 200 evaluation examples. Should retrieval be constrained by predicted/golden intent?
- **Decision Taken**: Vector retrieval is purely semantic nearest-neighbor search without intent filtering. Golden intent is used strictly for post-hoc analysis metrics.
- **Rationale & Evidence**:
  - Prevents data leakage and simulates real-world open-domain customer support where ground-truth labels do not exist at inference time.
  - Dense retrieval autonomously achieved **39.5% top-1 same-intent match** and **65.5% top-5 same-intent coverage** without explicit intent conditioning.

---

### Decision 19: Retrieval Diversity Measurement & Canned Response Monitoring
- **Context & Problem Statement**: Dense search can return five minor syntactic variants of the same generic "DM us" reply.
- **Decision Taken**: Implemented template diversity tracking and canned response frequency metrics.
- **Rationale & Evidence**:
  - Top-5 retrieval averaged **5.0 unique resolution templates per query** (100% of queries returned $\ge 3$ distinct templates; 0% returned 5 identical templates).
  - Generic deflection rate in retrieved top-1 cases was limited to only **4.5%**.

---

### Decision 20: Soft Reranking with Additive Intent Compatibility Bonus
- **Context & Problem Statement**: Pure semantic retrieval achieved 0.7280 similarity, but only 39.5% top-1 intent alignment because the retriever was unconditioned.
- **Decision Taken**: Implemented soft reranking with an additive compatibility bonus ($\beta = +0.10$) when historical case intent matches predicted intent.
- **Rationale & Evidence**:
  - Increased top-1 same-intent match rate from **39.5% to 50.0%** (+10.5% absolute gain, +26.58% relative gain) with a negligible change in raw similarity (0.7280 to 0.7244).
  - Selected $\beta = 0.10$ based on cross-validation on the development partition.

---

### Decision 21: Low-Confidence Fallback Guardrail
- **Context & Problem Statement**: If the intent classifier predicts an incorrect intent with low confidence, applying an intent bonus can boost an irrelevant case into top-1.
- **Decision Taken**: Gated reranking behind a minimum confidence threshold ($\tau_{\text{conf}} = 0.60$). If confidence $< 0.60$, the system bypasses the intent bonus and falls back cleanly to raw semantic nearest neighbors.
- **Rationale & Evidence**:
  - Safely protected 66.5% of ambiguous/low-confidence queries from intent poisoning, preserving dense lexical proximity.

---

### Decision 22: Candidate Pool Size ($k_{\text{candidate}} = 10$)
- **Context & Problem Statement**: Selecting the retrieval depth before reranking.
- **Decision Taken**: Set initial retrieval pool to $k_{\text{candidate}} = 10$, returning top-$k = 5$ post-reranking.
- **Rationale & Evidence**:
  - Top-10 vector retrieval takes $<1.5\text{ ms}$ on CPU while providing sufficient intent variety to promote intent-compatible historical cases without introducing distant semantic noise.

---

### Decision 23: Independent Evaluation of Intent Classifier Before Retrieval Reranking
- **Context & Problem Statement**: Ensuring classifier reproducibility and isolating classification error from retrieval error.
- **Decision Taken**: Evaluated the classifier independently on the golden set before evaluating the reranked pipeline.
- **Rationale & Evidence**:
  - Replicated Phase 5 baseline numbers exactly (Accuracy: 70.50%, Macro F1: 0.6915, Weighted F1: 0.7276), ensuring zero drift.

---

### Decision 24: Rejection of LLM-Based Reranking in Phase 7
- **Context & Problem Statement**: Could an LLM be used as a cross-encoder or reranker?
- **Decision Taken**: Explicitly chose deterministic algebraic reranking over an LLM.
- **Rationale & Evidence**:
  - Linear reranking executes in $<0.1\text{ ms}$ per query with 100% deterministic reproducibility and zero token cost.
  - Keeps RAG retrieval modular and inspectable before introducing the generative LLM in Phase 8.

---

### Decision 25: Explainable Component-Wise Scoring Schema
- **Context & Problem Statement**: Downstream agent debugging requires transparency into why a historical case ranked at position 1.
- **Decision Taken**: Standardized `RerankedResult` to expose `semantic_similarity`, `predicted_intent`, `historical_intent`, `intent_match`, `intent_bonus`, and `final_rerank_score`.
- **Rationale & Evidence**:
  - Gives complete explainability for safety audits and automated LLM prompt generation.

---

### Decision 26: Historical Responses Treated as Policy Evidence Rather Than Ground Truth
- **Context & Problem Statement**: Historical Twitter support responses reflect past interactions and frequently contain case-specific facts (order IDs, past dates, names). If an LLM treats them as ground truth facts for the current user, it creates severe hallucinations.
- **Decision Taken**: Prompts explicitly instruct the LLM that historical cases are examples of past resolution patterns and general self-service steps, NOT current customer facts or binding commitments.
- **Rationale & Evidence**:
  - Prevents the agent from hallucinating live order statuses or quoting stale 2017 timestamps.

---

### Decision 27: Provider Abstraction with Offline Deterministic Mock Mode
- **Context & Problem Statement**: The project must run reproducibly in CI, test environments, and during reviewer grading without requiring paid API credentials or experiencing non-deterministic API flakiness.
- **Decision Taken**: Built `LLMProvider` abstraction with `MockLLMProvider` (default), `GeminiLLMProvider`, and `OpenAILLMProvider`.
- **Rationale & Evidence**:
  - 100% of unit tests and offline evaluation scripts run seamlessly in deterministic mock mode with zero API keys.

---

### Decision 28: Deterministic Heuristic Formula for Evidence Quality Score
- **Context & Problem Statement**: Downstream triage routing needs to know if retrieval returned high-confidence, relevant evidence without calling a separate expensive LLM judge.
- **Decision Taken**: Implemented a transparent, deterministic heuristic formula:
  $$\text{Score} = (0.40 \cdot \text{Top1\_Sim}) + (0.30 \cdot \text{Mean\_Sim}) + (0.20 \cdot \text{Intent\_Match\_Ratio}) + (0.10 \cdot \text{Count\_Ratio}) - \text{Boilerplate\_Penalty}$$
- **Rationale & Evidence**:
  - Fully explainable, bounds scores to $[0.0, 1.0]$, and penalizes generic deflection boilerplate.

---

### Decision 29: Deterministic Grounding Guardrail for Live-Claim and PII Rejection
- **Context & Problem Statement**: Generative LLMs are prone to phrasing replies as if they checked internal systems ("I checked your account", "Your package was sent yesterday").
- **Decision Taken**: Implemented deterministic post-generation regex validation blocking forbidden live-system verbs, PII leaks, and 2017 historical timestamps.
- **Rationale & Evidence**:
  - Achieved 100% guardrail pass rate across all evaluated responses, rejecting unsafe generations.

---

### Decision 30: Priority of Safety and Risk Mitigation Over Headline Automation Rate
- **Context & Problem Statement**: Optimizing purely for a high `AUTO_HANDLE` rate results in dangerous failures (e.g. Phase 5 majority baseline had 100% auto-handle rate but 60/60 dangerous missed escalations).
- **Decision Taken**: Engineered the `TriageEngine` to be conservative, routing complex, sensitive, or low-confidence queries to human support.
- **Rationale & Evidence**:
  - Reduced dangerous false auto-handles by 45.8% compared to the Phase 5 TF-IDF baseline (13 vs 24) and 78.3% compared to the majority baseline (13 vs 60), while achieving 78.33% escalation recall.

---

### Decision 31: Mandatory Human Escalation for Account Security and Payment Disputes
- **Context & Problem Statement**: Certain customer problems (e.g., account lockouts, unauthorized card charges, missing delivered packages) inherently require authenticated human verification or carrier claims.
- **Decision Taken**: Hardcoded mandatory escalation rules triggering explicit reason codes (`ACCOUNT_SECURITY_ACTION`, `MISSING_PACKAGE_CLAIM`, `PAYMENT_DISPUTE_VERIFICATION`) regardless of model confidence.
- **Rationale & Evidence**:
  - Guarantees that sensitive security and financial claims are never dangerously deflected to automated canned responses.

---

### Decision 32: Five Core Dimensions for Response-Quality Evaluation
- **Context & Problem Statement**: General single-scalar scoring (e.g. 1-10 "quality") collapses distinct failure modes like factual hallucinations, inappropriate tone, or safety risks.
- **Decision Taken**: Decomposed evaluation into 5 distinct orthogonal dimensions: Correctness, Groundedness, Helpfulness, Safety, and Tone.
- **Rationale & Evidence**:
  - Enables granular diagnosis: an agent reply can be polite and well-toned (5/5) but ungrounded (2/5) or unsafe (1/5).

---

### Decision 33: Standard 1–5 Discrete Likert Scale with Anchored Behavioral Rubrics
- **Context & Problem Statement**: Open-ended numeric scales (e.g. 0-100) introduce massive judge variance and subjective drift.
- **Decision Taken**: Adopted a standardized 1-5 discrete scale with explicit semantic anchors for each level and explicit definitions for `critical_failure` and `acceptable`.
- **Rationale & Evidence**:
  - Maximizes inter-annotator agreement and provides strict criteria for automated LLM judges.

---

### Decision 34: 50-Example Stratified Human Evaluation Subset
- **Context & Problem Statement**: Evaluating all 200 examples by human review is labor-intensive, while a simple random 20-example sample misses critical edge cases.
- **Decision Taken**: Extracted a stratified 50-example subset balancing all 11 intents, routing decisions (AUTO_HANDLE vs ESCALATE), and boundary safety cases (seed=42).
- **Rationale & Evidence**:
  - Provides a statistically sound, representative benchmark for measuring LLM judge-to-human alignment.

---

### Decision 35: Strict Prohibition Against Fabricating Synthetic Human Scores
- **Context & Problem Statement**: When human annotations have not yet been manually submitted, systems often synthesize fake random human scores to fill in evaluation tables.
- **Decision Taken**: Explicitly reported `HUMAN AGREEMENT NOT YET MEASURED` and preserved ready-to-use annotation templates and calculation scripts.
- **Rationale & Evidence**:
  - Upholds scientific integrity and prevents misleading agreement claims.

---

### Decision 36: Decoupling System-Level Triage Metrics from Response-Quality Judge Metrics
- **Context & Problem Statement**: Conflating routing metrics (accuracy, dangerous false auto-handles) with generative quality (tone, empathy) creates an uninterpretable single metric.
- **Decision Taken**: Evaluated system-level triage metrics independently from generative LLM judge rubric scores.
- **Rationale & Evidence**:
  - Gives clear, isolated visibility into pipeline routing safety vs natural language generation quality.

---

### Decision 37: Quadratic Weighted Cohen's Kappa for Ordinal Metric Agreement
- **Context & Problem Statement**: Simple accuracy (exact match percentage) penalizes off-by-one differences (e.g., 4 vs 5) as harshly as complete contradictions (1 vs 5).
- **Decision Taken**: Implemented quadratic weighted Cohen's Kappa ($\kappa_w$) alongside Mean Absolute Difference (MAD) and Pearson correlation.
- **Rationale & Evidence**:
  - Standard established best practice in psychometrics and NLP evaluation for ordinal 1-5 ratings.

