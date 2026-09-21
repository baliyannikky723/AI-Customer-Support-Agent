# Dataset Schema & Structural Specification

This document provides the formal schema, column semantics, missing value distribution, and storage profiles for the **Customer Support on Twitter** dataset (`thoughtvector/customer-support-on-twitter`).

---

## 1. Dataset Files & Storage Overview

| Filename | Format | File Size | Row Count | Column Count | Primary Purpose |
|---|---|---|---|---|---|
| `twcs.csv` | CSV (UTF-8) | 492.58 MB | 2,811,774 | 7 | Complete corpus of multi-turn customer support interactions. |
| `sample.csv` | CSV (UTF-8) | 0.02 MB | 93 | 7 | Official minimal sample for rapid smoke-testing and schema verification. |
| `representative_sample.csv` | CSV (UTF-8) | 3.48 MB | 20,000 | 7 | Stratified representative sample generated for local development (`seed=42`). |

---

## 2. Comprehensive Field Schema & Semantics

| Column Name | Data Type | Null Count | Null % | Semantic Definition & Role | Example Values |
|---|---|---|---|---|---|
| `tweet_id` | `int64` | 0 | 0.00% | Unique global identifier for the tweet in Twitter's integer ID space. Primary key for node identification in conversation DAGs. | `119237`, `119238` |
| `author_id` | `string` | 0 | 0.00% | The author identity. For customer users, this is an anonymized numeric string. For support agents, this is the official brand handle string. | `'105834'`, `'AmazonHelp'`, `'AppleSupport'` |
| `inbound` | `bool` | 0 | 0.00% | Directionality indicator. `True` indicates a customer-to-brand message (incoming request). `False` indicates a brand-to-customer reply (outbound resolution). | `True`, `False` |
| `created_at` | `string` | 0 | 0.00% | RFC 2822 formatted timestamp indicating when the tweet was posted on Twitter. | `'Wed Oct 11 09:14:31 +0000 2017'` |
| `text` | `string` | 0 | 0.00% | Raw text content of the message, including user mentions (`@handle`), shortened URLs (`https://t.co/...`), emojis, and agent signatures. | `'@AmazonHelp my package was marked delivered but not received.'` |
| `response_tweet_id` | `string` | 1,040,629 | 37.01% | Forward graph edge(s). Comma-separated list of child tweet IDs that directly responded to this message. Null when the tweet is the terminal leaf of a thread. | `'119236'`, `'119240,119241'` |
| `in_response_to_tweet_id`| `float64`| 794,299 | 28.25% | Backward graph edge. The single parent tweet ID that this tweet was replying to. Null for root tweets (initial customer outreach or brand announcement). | `119242.0`, `NaN` |

---

## 3. High-Level Corpus Statistics

- **Total Messages**: 2,811,774
- **Customer Messages (`inbound == True`)**: 1,537,843 (54.69%)
- **Brand / Support Messages (`inbound == False`)**: 1,273,931 (45.31%)
- **Unique Active Brands**: 108 brands (with $\ge 10$ outbound responses)
- **Root Tweets (Opening Inquiries, `in_response_to == NaN`)**: 794,299 (28.25%)
- **Terminal Tweets (Closing Responses, `response_tweet_id == NaN`)**: 1,040,629 (37.01%)
- **Branching Multi-Responses (1 tweet triggering $\ge 2$ replies)**: 110,832 instances

---

## 4. Graph Topology & Linkage Semantics

The dataset is structured as a **Directed Acyclic Forest (DAF)** of conversation trees:
1. **Root Nodes (`in_response_to_tweet_id` is NaN)**: Initial customer complaints, questions, or brand public tweets.
2. **Directed Edges**:
   - Backward edge: `child.in_response_to_tweet_id == parent.tweet_id`
   - Forward edge: `parent.response_tweet_id` contains `child.tweet_id`
3. **Multi-Turn Traversal**: Any complete customer-support resolution exchange can be reconstructed by joining child tweets to their parent tweet using `in_response_to_tweet_id`.
