# Semantic Retrieval & Historical RAG Foundation Strategy

## 1. Overview & Objectives

In **Phase 6**, we replace the classical TF-IDF retrieval baseline from Phase 5 with a **dense semantic vector retrieval architecture** powered by `sentence-transformers/all-MiniLM-L6-v2` and FAISS (`IndexFlatIP`). 

The system enables grounded **Retrieval-Augmented Generation (RAG)** by fetching historical, human-resolved AmazonHelp customer service interactions that closely match incoming customer queries.

```
+-----------------------------------------------------------------------------------+
|                            DEVELOPMENT SPLIT (85%)                                |
|    55,476 English AmazonHelp Conversations (Strictly Pre-Cutoff: <= Nov 27 2017)   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        CORPUS PREPARATION & DEDUPLICATION                         |
|   - 8,000 Representative Customer-Support Interaction Pairs                       |
|   - PII Scrubbing: [CUSTOMER], [ORDER_ID], [URL], [EMAIL], [PHONE]                |
|   - Exact (Query, Response) Deduplication                                         |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                       DENSE VECTOR ENCODING & INDEXING                            |
|   - Embedding Model: sentence-transformers/all-MiniLM-L6-v2 (384 dims)            |
|   - L2 Unit Normalization (||v||_2 = 1.0)                                         |
|   - FAISS Index: IndexFlatIP (Exact Inner Product == Cosine Similarity)           |
|   - Serialized: data/processed/faiss_index.bin (11.72 MB)                         |
+-----------------------------------------------------------------------------------+
                                         |
                                         v (Top-k=5 Nearest Neighbors)
+-----------------------------------------------------------------------------------+
|                     HELD-OUT GOLDEN EVALUATION (200 QUERIES)                      |
|   - Strictly Post-Cutoff (>= Nov 27 2017 11:32:57 UTC)                            |
|   - Zero Overlap in Conversation IDs, Text, or Timestamps                         |
+-----------------------------------------------------------------------------------+
```

---

## 2. Embedding Model Selection

| Criterion | `all-MiniLM-L6-v2` | Alternatives (`bge-small-en`, `e5-small`) | Rationale |
| :--- | :--- | :--- | :--- |
| **Parameters** | 22.7 Million | 33M - 110M | Extreme CPU inference efficiency |
| **Embedding Dimension** | 384 | 384 - 768 | 50% smaller memory footprint than 768-dim models |
| **Latency per batch (64)** | ~0.55 seconds (CPU) | 1.2s - 3.4s | Real-time support query response (<15ms per single query) |
| **Reproducibility** | Universal PyPI / HuggingFace support | Requires custom query prefixes | Deterministic behavior without prompt engineering |
| **Sequence Length** | 256 tokens | 512 tokens | Twitter customer inquiries average 28.4 tokens (max 70 tokens) |

---

## 3. FAISS Index Configuration

We chose **`faiss.IndexFlatIP`** (Exact Inner Product Search) paired with **L2-normalized embeddings**:

$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \mathbf{u}_{\text{norm}} \cdot \mathbf{v}_{\text{norm}} = \text{Inner Product}(\mathbf{u}_{\text{norm}}, \mathbf{v}_{\text{norm}})$$

### Rationale for Exact Search over Approximate (ANN) Indexing:
1. **Mathematical Exactness**: Guarantees true nearest-neighbor retrieval without recall degradation ($100\%$ precision at top-$k$).
2. **Corpus Scale**: For $N=8,000$ to $55,000$ vectors in 384 dimensions, brute-force matrix multiplication takes $<1.5\text{ ms}$ on CPU. Approximate indexing (e.g., IVF-PQ, HNSW) adds quantization error, index construction complexity, and hyperparameter tuning overhead with zero noticeable latency advantage at this scale.
3. **Reproducibility**: Eliminates clustering centroid variance across operating systems.

---

## 4. Query Representation Strategy

The query representation determines how the incoming customer state is converted into an embedding:

1. **`current_only` (Default)**:
   - Encodes the current customer message: `query_text = current_customer_message`.
   - *Why Default*: In customer support on Twitter, initial customer messages are highly descriptive (e.g., *'My package was marked delivered but never arrived'*). Including previous bot messages or generic greetings dilutes the core issue embedding with non-informative conversational filler.
2. **`context_plus_current` (Configurable)**:
   - Concatenates the preceding customer turn if available: `query_text = previous_customer_message + " [SEP] " + current_customer_message`.
   - Useful for multi-turn follow-up queries where the customer provides supplementary details.

---

## 5. Corpus Construction & Deduplication

1. **Temporal Isolation**: Sourced exclusively from the pre-cutoff development partition ($N=55,476$ conversations prior to `2017-11-27 11:32:57 UTC`).
2. **PII Masking**: All historical responses are pre-sanitized with `PIISanitizer` to ensure no customer Twitter handles (`@\d+`), Amazon order numbers (`\d{3}-\d{7}-\d{7}`), phone numbers, or tracking URLs are exposed in retrieved contexts.
3. **Deduplication Policy**:
   - Exact `(customer_query, amazon_response)` duplicates are eliminated to avoid redundant index slots.
   - Distinct queries that share high-level resolution policies (e.g., standard replacement procedures) are preserved to maintain genuine historical resolution frequencies.

---

## 6. Performance & Operational Benchmarks

- **Corpus Size**: 8,000 historical support interactions
- **Vector Dimension**: 384 float32
- **FAISS Binary File Size**: 11.72 MB (`data/processed/faiss_index.bin`)
- **Metadata Parquet File Size**: 1.23 MB (`data/processed/resolution_metadata.parquet`)
- **Index Build Throughput**: 113.4 cases/sec on standard CPU (70.54s total encoding time)
- **Query Retrieval Latency**: 2.3 ms per query ($k=5$)
- **Golden Evaluation Time (200 queries)**: 0.46 seconds total
