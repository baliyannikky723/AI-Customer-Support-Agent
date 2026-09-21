# AmazonHelp Response Patterns & Canned Template Analysis

This report documents the structural patterns, repetition frequency, and linguistic distribution of AmazonHelp support responses.

---

## 1. Corpus Linguistic Breakdown

Using deterministic multi-script and stopword analysis across all 367,148 cleaned AmazonHelp messages:

| Language Category | Message Count | Percentage of Corpus | Notes & Distribution |
|---|---|---|---|
| **English (`en`)** | **289,902** | **78.96%** | Primary language; substantial volume for development, retrieval, and evaluation. |
| **Spanish (`es`)** | 23,472 | 6.39% | Handled by Amazon ES/MX regional teams. |
| **Japanese (`ja`)** | 19,700 | 5.37% | Distinct Hiragana/Katakana/Kanji script (Amazon JP). |
| **French (`fr`)** | 14,527 | 3.96% | Handled by Amazon FR/CA teams. |
| **German (`de`)** | 7,846 | 2.14% | Handled by Amazon DE team. |
| **Ambiguous / Short** | 7,401 | 2.02% | Very short phrases (`"OK"`, `"Thanks"`, single emojis). |
| **Portuguese (`pt`)** | 4,119 | 1.12% | Handled by Amazon BR team. |
| **Other (Hindi, Arabic, Russian)**| 181 | 0.05% | Minority regional inquiries. |

### Strategic Recommendation for Project Scope
**Recommendation: `language.mode = "english_only"` (with configurable preprocessing switch)**.
- **Rationale**: English constitutes ~79% of the corpus (~290,000 messages and ~122,000 complete English conversation pairs). Restricting the development and golden evaluation set to English ensures high semantic consistency in intent classification, prevents multilingual retrieval confusion in embeddings, and aligns with standard LLM-as-a-judge prompt rubrics.
- **Configurability**: Implemented via `configs/default_config.yaml` (`language.mode: english_only`), allowing future expansion to multilingual without code changes.

---

## 2. Repeated Support Response & Canned Template Analysis

| Metric | Raw Text (with Handles & Signatures) | Sanitized Text (Handles & Sign-offs Masked) |
|---|---|---|
| **Total Support Messages** | 169,839 | 169,839 |
| **Unique Text Variations** | 169,805 (99.98%) | 152,627 (89.87%) |
| **Exact Template Duplication Rate**| 0.02% | **10.13%** (17,212 instances) |

### Key Insight
At the raw string level, almost every tweet appears unique ($99.98\%$) because Amazon support agents prepend the customer's unique numerical handle (`@105834`) and append unique employee signatures (`^SN`, `^MM`). Once PII and signatures are normalized, the underlying policy templates emerge.

---

## 3. Top Repeated Support Templates

1. **Public PII Warning Template** (~1,500 cumulative variations):
   > `"[CUSTOMER] Please don't provide your order details, we consider it personal information. Please reach out to us securely here: [URL]"`
2. **Delivery Issue Escalation Template** (~850 variations):
   > `"[CUSTOMER] Sorry for the hassle. Please report this to our support team here: [URL] and we'll investigate right away."`
3. **Account / Payment Security Guidance** (~600 variations):
   > `"[CUSTOMER] We'd like to look into this for you. Please contact us via chat or phone at [URL] so we can verify your account securely."`
4. **Carrier / Tracking Clarification Template** (~450 variations):
   > `"[CUSTOMER] Has the estimated delivery date passed? You can track the latest carrier scans in Your Orders: [URL]"`

---

## 4. Retrieval Deduplication & RAG Strategy

If raw resolution pairs were embedded directly into a vector database without deduplication, nearest-neighbor search for any delivery complaint would retrieve 10 near-identical *"Please reach out securely at [URL]"* strings.

**Mitigation in RAG Pipeline**:
1. **Template Clustering & Frequency Tracking**: Keep representative resolution instances in the vector store while tracking how frequently each policy action is applied.
2. **Context Enrichment**: Index the combined `(Customer Inquiry, Amazon Contextual Resolution)` rather than standalone canned replies, ensuring retrieval is conditioned on the customer's specific problem symptoms.
