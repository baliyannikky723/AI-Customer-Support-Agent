# Candidate Brand Evaluation & Comparative Analysis

This report evaluates 5 candidate brands from the Twitter Customer Support dataset across quantitative availability, problem diversity, resolution quality, and suitability for building and evaluating an AI support agent.

---

## 1. Factual Brand Comparison Matrix

| Evaluation Dimension | Candidate 1: `AmazonHelp` | Candidate 2: `AppleSupport` | Candidate 3: `Uber_Support` | Candidate 4: `SpotifyCares` | Candidate 5: `Delta` |
|---|---|---|---|---|---|
| **Total Available Messages** | **404,889** | 240,352 | 124,842 | 89,589 | 101,161 |
| **Support Responses** | 169,840 | 106,860 | 56,270 | 43,265 | 42,253 |
| **Usable Conversation Pairs** | **155,445** | 106,696 | 55,283 | 41,734 | 36,215 |
| **Problem Domain Breadth** | High (Shipping, Returns, Prime, Digital, Payments, Refunds) | Moderate-High (iOS, Battery, iCloud, Mac, Screen, Subscriptions) | Moderate (Fares, Lost Items, Driver Safety, Cancellations, UberEats) | Focused (Music Playback, Offline Sync, Premium Billing, Login) | High-Urgency (Flight Delays, Baggage, Rebooking, Upgrades, Seats) |
| **Resolution Substantiveness** | High (Direct policy steps + targeted self-serve URLs) | High (Diagnostic questions + support article links) | Moderate (High proportion of in-app help/DM requests) | High (Direct troubleshooting + settings guidance) | Moderate-Low (High DM requirement for flight PII / seat privacy) |
| **DM Deflection Rate** | **0.6%** (Very low initial deflection) | 46.8% | 35.6% | 27.0% | 14.4% |
| **Link / Resource Provision Rate** | 37.9% | **65.5%** | 48.4% | 46.2% | 13.1% |
| **Agent Sign-off Regularity** | 92.3% (`^XX` pattern) | 0.0% | 0.0% | 0.0% | 0.1% |
| **Non-English Language Share** | 22.2% (Requires `en` filter) | 5.7% | 0.6% | 21.1% | 10.2% |
| **Intent Distinction Clarity** | Clear, distinct operational intents | Technical troubleshooting overlaps | Clear operational ride/fare intents | Very narrow technical intents | High emotional urgency / escalation skew |
| **Golden Eval Set Feasibility (150–250)** | **Extremely High** (Massive variety of clear scenarios) | **High** (Well-structured technical cases) | **High** (Rich incident reports) | **Moderate** (Risk of repetitive cases) | **Moderate** (Heavily skewed towards flight status) |

---

## 2. Detailed Profile & Trade-offs per Candidate

### Candidate 1: `AmazonHelp` (E-Commerce & Retail Support)
- **Strengths**:
  - Largest volume in the dataset (155,445 usable pairs), ensuring ample retrieval depth.
  - Extremely low immediate DM deflection (0.6%), meaning historical replies contain actual substantive guidance, instructions, and self-serve URLs rather than generic *"Please DM us"*.
  - Diverse, well-defined intent taxonomy (Order Tracking, Refund Request, Return Procedure, Prime Membership, Damaged Item, Digital Content).
  - Clear escalation boundaries (e.g. account takeover, unauthorized charges, repeated lost packages require human escalation).
- **Trade-offs**:
  - ~22% non-English tweets require language filtering during data preprocessing.
  - Frequent PII (order IDs) requires strict sanitization.

---

### Candidate 2: `AppleSupport` (Consumer Electronics & Software)
- **Strengths**:
  - High volume (106,696 usable pairs) with excellent link provision (65.5% direct troubleshooting links).
  - Low non-English presence (5.7%).
  - Rich multi-turn technical troubleshooting dialogues.
- **Trade-offs**:
  - 46.8% of replies immediately request DMs or device serial numbers.
  - Subtle technical differences (e.g. iOS 11 vs iOS 10 bugs) can make intent boundaries fuzzier for automated classification.

---

### Candidate 3: `Uber_Support` (Mobility & Gig Economy)
- **Strengths**:
  - Very clean English corpus (99.4% ASCII).
  - High operational realism with clear intent categories: Fare Dispute, Lost Item in Vehicle, Driver Conduct/Safety, Cancellation Fee, Food Delivery.
  - Excellent escalation test cases (driver safety / assault / accidents are mandatory immediate human escalations).
- **Trade-offs**:
  - Relatively high proportion of canned deflection to in-app help menus.

---

### Candidate 4: `SpotifyCares` (Digital Media & Streaming)
- **Strengths**:
  - Highly structured technical problem domain (offline downloads, Family plan billing, desktop app crashes).
  - Friendly, distinct brand tone.
- **Trade-offs**:
  - Narrower domain scope: customer queries can become repetitive, reducing intent diversity.
  - 21.1% multilingual content.

---

### Candidate 5: `Delta` (Airlines & Travel)
- **Strengths**:
  - High stakes: clear contrast between auto-handle (flight status, baggage allowance) and escalation (cancelled flight, missed connection, medical emergency).
- **Trade-offs**:
  - Support agents rarely resolve rebooking in public tweets due to PII/passenger privacy, leading to many short deflection tweets.

---

## 3. Evaluation Rigor & Feasibility Summary

| Candidate | Intent Clarity | RAG Groundedness Potential | Escalation Policy Contrast | Evaluation Set Construction Ease |
|---|---|---|---|---|
| **`AmazonHelp`** | ⭐⭐⭐⭐⭐ (5/5) | ⭐⭐⭐⭐⭐ (5/5) | ⭐⭐⭐⭐⭐ (5/5) | ⭐⭐⭐⭐⭐ (5/5) |
| **`AppleSupport`** | ⭐⭐⭐⭐ (4/5) | ⭐⭐⭐⭐ (4/5) | ⭐⭐⭐⭐ (4/5) | ⭐⭐⭐⭐ (4/5) |
| **`Uber_Support`** | ⭐⭐⭐⭐ (4/5) | ⭐⭐⭐ (3/5) | ⭐⭐⭐⭐⭐ (5/5) | ⭐⭐⭐⭐ (4/5) |
| **`SpotifyCares`** | ⭐⭐⭐ (3/5) | ⭐⭐⭐⭐ (4/5) | ⭐⭐⭐ (3/5) | ⭐⭐⭐ (3/5) |
| **`Delta`** | ⭐⭐⭐ (3/5) | ⭐⭐ (2/5) | ⭐⭐⭐⭐⭐ (5/5) | ⭐⭐⭐ (3/5) |

**Note**: This report presents empirical data and trade-offs. The final brand selection will be confirmed explicitly prior to Phase 2.
