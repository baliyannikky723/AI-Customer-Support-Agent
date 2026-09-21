# Data Split & Train/Evaluation Leakage-Prevention Strategy

This document establishes the formal partition architecture designed to ensure zero data leakage between development/retrieval corpora and future evaluation sets.

---

## 1. Core Principles of Leakage Prevention

In Retrieval-Augmented Generation (RAG) and Intent Classification systems, data leakage occurs when:
1. **Self-Retrieval**: An evaluation query retrieves its own historical resolution from the vector index.
2. **Near-Duplicate Overlap**: Paraphrased versions of the same customer issue authored by the same user or in the same conversation thread exist in both retrieval and test sets.
3. **Temporal Lookahead**: Training or indexing on future data to answer past customer queries.

---

## 2. Chronological (Time-Aware) Partitioning

Conversations in the AmazonHelp corpus are ordered by initial timestamp (`created_at`):

```
┌────────────────────────────────────────────────────────┬─────────────────────────────┐
│             Development & Retrieval Corpus             │    Golden Evaluation Set    │
│                       (~85% Volume)                    │        (~15% Volume)        │
│          Earliest Date  ──────────►  Cutoff Date       │ Cutoff Date ──────► Latest  │
└────────────────────────────────────────────────────────┴─────────────────────────────┘
```

- **Development / RAG Retrieval Corpus (Train/Dev)**:
  - Formed from the earlier 85% chronological partition of complete conversations.
  - Used for FAISS vector index construction, BM25 keyword matching, and few-shot intent exemplar selection.
- **Golden Evaluation Set Candidate Pool**:
  - Sampled strictly from the final 15% chronological partition.
  - Guarantees true out-of-time evaluation modeling how the agent performs on future customer queries.

---

## 3. Strict Isolation Rules & Verification

| Guardrail Layer | Enforcement Mechanism | Verification Script |
|---|---|---|
| **Conversation ID Isolation** | Hard disjoint set check: $\text{IDs}_{\text{eval}} \cap \text{IDs}_{\text{retrieval}} = \emptyset$ | `tests/test_split_isolation.py` |
| **Author ID Isolation** | Exclude repeat customer threads across partition boundaries. | Preprocessing partition validator. |
| **Exact Query Deduplication** | Ensure no normalized customer query in the eval set has an exact string match in the retrieval index. | Levenshtein & exact match filters. |
| **Vector Index Masking** | When running full-agent evaluation, verify that the active FAISS index excludes all eval set IDs. | Evaluation harness assertions. |

---

## 4. Addressing Canned Response Overlap

Because Amazon support agents apply standard policy templates (e.g. directing customers to `[URL]` for account verification), canned advice can appear across both partitions.

**Resolution**:
- Evaluation metrics will evaluate whether the agent selected the **correct domain-specific policy** for the customer's specific intent (e.g., Return Policy vs. Carrier Delay Policy) rather than verbatim lexical overlap (BLEU).
- LLM-as-a-judge rubric evaluates **relevance, factual grounding against retrieved policy, and escalation reasoning**, avoiding bias towards memorized text.
