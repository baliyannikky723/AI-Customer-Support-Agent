# Data Quality, Anomaly, & Leakage Risk Analysis

This report documents the empirical data-quality audit conducted on the 2.81M tweet dataset, highlighting edge cases, PII risks, and mitigation strategies for our AI agent pipeline.

---

## 1. Summary of Data Quality Metrics

| Anomaly / Quality Dimension | Estimated Frequency | Example Pattern | Project Impact | Recommended Handling |
|---|---|---|---|---|
| **Missing Backward Link (`in_response_to_tweet_id`)** | 28.25% (794k msgs) | Root customer inquiries or brand broadcasts | Normal DAG structure | Identify as potential thread roots. |
| **Missing Forward Link (`response_tweet_id`)** | 37.01% (1.04M msgs)| Terminal resolution tweets or unresponded queries | Normal DAG structure | Identify thread termination points. |
| **Extremely Short Messages ($< 10$ chars)** | 2.14% (60k msgs) | `"DM sent"`, `"help"`, `"thanks"`, `"why?"` | Low semantic signal for intent classification | Filter out non-actionable acknowledgments from golden eval set. |
| **Agent Sign-off Noise (`^XX`, `-Name`)** | ~92% (AmazonHelp), ~0% (AppleSupport) | `"Please reach out here: ... ^SN"` | Distorts RAG embedding vectors | Strip regex `\^[A-Z]{2,4}` during text normalization. |
| **Shortened Tracking URLs (`https://t.co/...`)** | 38.7% (Brand replies)| `"Check details at https://t.co/xyz..."` | Critical for self-serve grounding | Preserve URL token positions; map domain destinations where possible. |
| **Exact Duplicate Tweets** | 3.4% of total text | Identical canned support replies (`"Please DM us..."` repeated thousands of times) | Vector index clustering & RAG retrieval collapse | Deduplicate historical resolution index so RAG retrieves diverse resolution strategies. |
| **Non-English / Multilingual Text** | 22.2% (AmazonHelp), 5.7% (AppleSupport) | Spanish, Japanese, German, Portuguese support | Multi-lingual confusion in intent classification | Filter dataset to English (`en`) during brand indexing. |
| **PII in Inbound Customer Queries** | ~4.8% of inbound queries | Order IDs (`112-3456789-1234567`), phone numbers, email addresses | Privacy violation & model memorization | Implement automated regex PII scrubber for order IDs, emails, and phone numbers. |
| **Data Leakage Risk (Train / Eval Contamination)** | High if randomly split | Identical or near-identical customer queries in both Retrieval index and Golden Eval set | Artificially inflated RAG and classification accuracy | **Strict Time-Based or Conversation-Based Split**: Golden set must be strictly isolated from RAG vector corpus. |

---

## 2. Detailed Breakdown of Key Quality Hazards

### A. PII and Sensitive Information Exposure
While Twitter user IDs are numeric hashes in this dataset, customer tweets frequently contain raw unmasked sensitive data:
- **Amazon Order Numbers**: Format `\d{3}-\d{7}-\d{7}`
- **Email Addresses**: Format `[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+`
- **Phone Numbers**: 10-digit international / US phone strings
- **Tracking Numbers**: UPS / FedEx tracking strings

**Mitigation**: The preprocessing pipeline (`src/preprocessing/sanitizer.py`) will automatically mask PII with standardized tokens (`<ORDER_ID>`, `<EMAIL>`, `<PHONE>`, `<URL>`).

### B. The "Canned Response" Collapse
A significant fraction of brand replies are generic deflection templates:
- *"Please send us a DM with your account details and order number so we can look into this."*
If the vector index contains 50,000 identical DM requests, RAG retrieval will always retrieve a generic deflection instead of specific diagnostic and resolution instructions.

**Mitigation**: Cluster and deduplicate resolution templates before indexing into FAISS, ensuring high-entropy, solution-rich examples are prioritized.

### C. Data Leakage & Evaluation Rigor
If the retrieval index contains the exact historical answer to a golden set customer tweet, the system achieves an artificial 100% retrieval match that does not reflect real-world generalization.

**Mitigation**:
1. All golden evaluation examples will be **completely removed** from the FAISS retrieval index.
2. Temporal or author-level isolation will prevent near-neighbor leakage.
