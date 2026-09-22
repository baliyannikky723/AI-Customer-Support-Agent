# Intent-Aware Historical Retrieval & Reranking Strategy

## 1. Executive Overview & Problem Context

In **Phase 6**, pure dense semantic retrieval with `all-MiniLM-L6-v2` achieved a high mean cosine similarity (`0.7280`), but was **unconditioned**: only **39.5%** of golden queries retrieved a historical case matching the customer's true operational intent at rank 1.

In **Phase 7**, we designed and integrated an **Intent-Aware Candidate Reranking System** that bridges classification and retrieval. By conditioning the vector candidate pool on classifier predictions with an explainable compatibility bonus and confidence gate, top-1 intent alignment increases from **39.5% to 50.0%** with zero data leakage.

```
                                  [Incoming Customer Message]
                                               │
                                               ▼
                                ┌───────────────────────────────┐
                                │    PII Sanitizer & Scrubber   │
                                └──────────────┬────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
        ┌─────────────────────────────┐                 ┌─────────────────────────────┐
        │   TF-IDF Intent Classifier  │                 │    FAISS Vector Retriever   │
        │  - Predicted Intent         │                 │  - candidate_k = 10         │
        │  - Confidence Score [0, 1]  │                 │  - Exact Cosine Similarity  │
        │  - Top-3 Distribution       │                 │  - Normalized MiniLM (384d) │
        └──────────────┬──────────────┘                 └──────────────┬──────────────┘
                       │                                               │
                       │ {pred_intent, confidence}                     │ [10 historical candidates]
                       └───────────────────────┬───────────────────────┘
                                               │
                                               ▼
                                ┌───────────────────────────────┐
                                │     Intent-Aware Reranker     │
                                │                               │
                                │ IF confidence < 0.60:         │
                                │    -> Fallback to raw sim     │
                                │ ELSE:                         │
                                │    score = sim + (bonus if    │
                                │            match else 0.0)    │
                                └──────────────┬────────────────┘
                                               │
                                               ▼ (Top-k=5 Reranked)
                                ┌───────────────────────────────┐
                                │  Explainable Retrieval Bundle │
                                │  {sim, match, bonus, score}   │
                                └───────────────────────────────┘
```

---

## 2. Intent-Aware Reranking Formula & Scoring Dynamics

### Mathematical Formulation
For a customer query $q$ with predicted intent $\hat{y}$ and classifier confidence $c \in [0, 1]$, and a retrieved historical case $r_i$ with semantic cosine similarity $s_i \in [-1, 1]$ and historical intent metadata $y_i$:

$$\text{RerankScore}(r_i, q) = \begin{cases} s_i, & \text{if } c < \tau_{\text{conf}} \text{ (Low-Confidence Fallback)} \\ s_i + \beta \cdot \mathbb{I}(y_i = \hat{y}), & \text{if } c \ge \tau_{\text{conf}} \text{ (Soft Reranking)} \end{cases}$$

Where:
- $\beta = 0.10$: Configurable additive intent compatibility bonus (`intent_match_bonus`).
- $\tau_{\text{conf}} = 0.60$: Confidence gating threshold (`min_confidence`).
- $\mathbb{I}(y_i = \hat{y}) \in \{0, 1\}$: Binary indicator of intent compatibility.

---

## 3. Development Tuning Procedure (Zero Golden Leakage)

To select the hyperparameter tuple $(\beta, \tau_{\text{conf}}, k_{\text{candidate}})$, we conducted a 5-fold cross-validation grid search strictly on the **development partition (8,000 cases)**:

| Candidate Bonus ($\beta$) | Confidence Gate ($\tau$) | Candidate Pool ($k_{\text{cand}}$) | Dev Same-Intent Top-1 | Mean Raw Sim | Selection Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0.00` (Raw Semantic) | `N/A` | 10 | 38.2% | 0.7291 | Unconditioned baseline. |
| `0.05` | 0.50 | 10 | 44.1% | 0.7265 | Bonus too weak to promote relevant matches past rank 2. |
| **`0.10` (Selected)** | **0.60** | **10** | **49.8%** | **0.7242** | **Optimal Pareto frontier: +11.6% intent gain with only -0.0049 sim drop.** |
| `0.15` | 0.60 | 10 | 51.2% | 0.7180 | Over-biases towards superficial intent metadata over semantic relevance. |
| `0.20` | 0.70 | 10 | 52.0% | 0.7090 | Excessive similarity degradation; promotes off-topic matches sharing broad tags. |

**Selected Configuration**:
- `intent_match_bonus: 0.10`
- `min_confidence: 0.60`
- `candidate_k: 10`
- `top_k: 5`

---

## 4. Architectural Decisions & Trade-offs

### A. Why Soft Reranking is Preferred Over Hard Filtering
1. **Historical Metadata Noise**: Support interactions can be context-dependent or multi-intent (e.g. damaged goods combined with refund inquiry).
2. **Recall Preservation**: Hard filtering (`mode: hard_intent_filter`) completely discards semantically brilliant historical cases if their historical rule tag differed slightly.
3. **Graceful Degradation**: Soft reranking allows an exceptionally high semantic match ($s_i = 0.88$) to outrank an intent-matching but semantically generic candidate ($s_j = 0.65 + 0.10 = 0.75$).

### B. Why Low-Confidence Fallback is Critical
- If classifier confidence is low ($<0.60$), the prediction is ambiguous (e.g., short queries like *"where is it"* or multi-intent grievances).
- Applying an intent bonus under high uncertainty risks boosting an irrelevant case into top-1.
- Gating reranking with $\tau_{\text{conf}} = 0.60$ guarantees that uncertain queries fall back cleanly to raw semantic nearest neighbors.

### C. Candidate Pool Size ($k_{\text{candidate}} = 10$)
- FAISS retrieves 10 nearest neighbors in $<1.5\text{ ms}$.
- Searching 10 candidates provides sufficient intent diversity to locate at least one intent-aligned historical record without introducing distant semantic noise.

---

## 5. Critical Analysis: What Can Still Go Wrong When Intents Match?

> [!WARNING]
> **Key Grounding Failure Modes**: Even when `predicted_intent == historical_intent`, the retrieved resolution **must not be blindly copied by an agent**:
>
> 1. **Temporal Context Staleness**: A retrieved refund case states *"Your refund was issued on Nov 15"*. The new customer needs general policy turnaround (5-7 business days), not historical dates.
> 2. **Conditional Policy Branches**: A return case states *"Return label generated"*. The new customer's item may be past the 30-day window or subject to hygiene exemptions.
> 3. **Geographic / Carrier Discrepancy**: A delivery inquiry retrieves a resolution citing *Royal Mail (UK)*, whereas the incoming customer is located in the *US (USPS/UPS)*.
>
> This demonstrates why **Phase 8 (LLM Response Generator & Triage Engine)** is essential: it uses retrieved cases for factual policy principles while dynamically adapting to the user's specific context.
