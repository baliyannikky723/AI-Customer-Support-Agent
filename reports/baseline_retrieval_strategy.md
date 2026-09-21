# Baseline Retrieval & Deduplication Strategy Report

This document details the corpus construction, template deduplication rules, and similarity scoring methodology for the classical TF-IDF retrieval baseline.

---

## 1. Retrieval Corpus Assembly & Leakage Guardrails

The historical resolution corpus used by the TF-IDF retriever is constructed strictly from the **Development Partition (Earliest 85% of timeline)**:
- **Source Dataset**: `data/processed/amazonhelp_conversations.parquet` (pre-cutoff: $\le \text{2017-11-27 11:30:51}$).
- **Leakage Constraint**: **Zero golden evaluation examples** ($N=200$) exist in the retrieval index. Verified via automated disjoint ID set checks.
- **Privacy Sanitization**: All historical customer queries and support responses are scrubbed using `PIISanitizer` to replace order numbers, emails, phone numbers, and customer handles with standardized tokens (`[ORDER_ID]`, `[EMAIL]`, `[PHONE]`, `[CUSTOMER]`, `[URL]`).

---

## 2. Template Deduplication Strategy

Phase 2 established that $10.13\%$ of raw support messages are identical canned templates once handles and employee signatures are stripped. Naively indexing 50,000 instances of *"Please reach out securely at [URL]"* causes nearest-neighbor search to collapse into generic deflection replies.

### Deduplication Pipeline:
1. **Normalized Query Clustering**: Collapse near-identical customer queries authored by the same user or within the same thread.
2. **Representative Selection**: For clusters sharing identical sanitized customer queries and response templates, preserve one high-quality representative instance while recording frequency metadata.
3. **Entropy Filtering**: Filter out near-empty queries ($<10$ characters) to ensure the TF-IDF vector space is rich in domain terms.

---

## 3. Retrieval Scoring & Query Transformation

- **Vector Representation**: Sublinear term frequency TF-IDF ($1+\log(\text{tf})$) with unigrams and bigrams ($1, 2$) over a maximum vocabulary of 10,000 features.
- **Similarity Metric**: Cosine similarity:
$$\text{Sim}(q, d) = \frac{\mathbf{v}_q \cdot \mathbf{v}_d}{\|\mathbf{v}_q\|_2 \|\mathbf{v}_d\|_2}$$
- **Output Output Contract**:
  - `retrieved_case_id`: Identifier of the matched historical conversation.
  - `similarity_score`: Float between $0.0$ and $1.0$.
  - `retrieved_customer_query`: Historical sanitized inquiry.
  - `retrieved_historical_response`: Historical AmazonHelp resolution.
