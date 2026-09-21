# AmazonHelp Data Extraction & Conservative Cleaning Report

This report documents the filtering and cleaning pipeline applied to isolate the AmazonHelp customer support corpus from the raw 2.81M tweet dataset.

---

## 1. Pipeline Execution Overview

- **Source Dataset**: `twcs.csv` (2,811,774 total rows across all brands)
- **Target Brand**: `AmazonHelp`
- **Output Corpus**: `data/processed/amazonhelp_raw.parquet`
- **Execution Time**: 37.29 seconds

---

## 2. Before & After Cleaning Statistics

| Cleaning Stage / Filter Rule | Records Before | Records After | Records Removed / Filtered | % Impact |
|---|---|---|---|---|
| **Raw AmazonHelp Extraction** (Outbound + Inbound mentions & parents) | 2,811,774 | 367,149 | 2,444,625 (non-Amazon) | 86.8% (Target Filtering) |
| **Duplicate `tweet_id` Removal** | 367,149 | 367,149 | 0 | 0.000% |
| **Empty / Whitespace Text Removal** | 367,149 | 367,149 | 0 | 0.000% |
| **Exact Content Duplicate Removal** (`author_id` + `text` + `created_at`) | 367,149 | 367,148 | 1 | 0.000% |
| **Malformed Timestamp Filter** | 367,148 | 367,148 | 0 | 0.000% |
| **Final Cleaned AmazonHelp Corpus** | — | **367,148** | — | — |

---

## 3. Composition of the Cleaned AmazonHelp Corpus

- **Total Cleaned Records**: **367,148**
- **Support Messages (`inbound == False`, author: `AmazonHelp`)**: **169,839** (46.26%)
- **Customer Messages (`inbound == True`)**: **197,309** (53.74%)
- **Customer / Support Ratio**: 1.16 : 1.00

---

## 4. Text Preservation Integrity

The following key text features were strictly preserved without destructive normalization:
1. **Punctuation & Sentences**: All exclamation marks, question marks, and multi-sentence structures retained for intent classification and tone modeling.
2. **URLs & Hyperlinks**: Retained in full for subsequent domain extraction and self-serve grounding.
3. **Emojis & Unicode**: Preserved to capture emotional urgency, frustration, and sentiment.
4. **Order Identifiers & Numbers**: Retained in raw format for targeted PII sanitization in Step 4.
