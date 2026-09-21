# Baseline Systems Evaluation Summary Report

This report documents the performance of the two required non-LLM baseline systems on the 200-example Golden Evaluation Set.

---

## 1. Executive Summary & Benchmark Comparison

| Evaluation Metric | Baseline 1 (Majority Class) | Baseline 2 (TF-IDF + Logistic Regression) |
|---|---|---|
| **Intent Classification Accuracy** | **4.00%** | **70.50%** |
| **Intent Macro F1 Score** | **0.0070** (0.024) | **0.6915** |
| **Intent Weighted F1 Score** | **0.0031** | **0.7276** |
| **Triage Handling Accuracy** | **70.00%** | **63.50%** |
| **Triage Escalation F1** | **0.0000** (0.000) | **0.4966** |
| **Dangerous False AUTO_HANDLE Count** | **60 / 60** (100% missed) | **24 / 60** |

---

## 2. Per-Intent Performance Breakdown (TF-IDF Baseline)

| Intent Name | Support ($N=200$) | Precision | Recall | F1 Score |
|---|---|---|---|---|
| `account_access_and_security` | 10 | 1.0000 | 0.6000 | **0.7500** |
| `damaged_defective_or_wrong_item` | 20 | 0.9375 | 0.7500 | **0.8333** |
| `delivery_status_tracking` | 28 | 0.9091 | 0.7143 | **0.8000** |
| `late_delivery_complaint` | 32 | 0.9000 | 0.8438 | **0.8710** |
| `order_cancellation_request` | 13 | 0.7000 | 0.5385 | **0.6087** |
| `order_delivered_not_received` | 16 | 1.0000 | 0.3125 | **0.4762** |
| `other_unknown` | 8 | 0.2222 | 1.0000 | **0.3636** |
| `payment_and_billing_issues` | 15 | 1.0000 | 0.8000 | **0.8889** |
| `prime_membership_and_digital` | 14 | 0.5238 | 0.7857 | **0.6286** |
| `refund_status_and_request` | 22 | 0.6429 | 0.8182 | **0.7200** |
| `return_and_pickup_inquiry` | 22 | 0.8571 | 0.5455 | **0.6667** |

---

## 3. Triage & Escalation Analysis (The Critical Safety Frontier)

- **Total Ground-Truth Escalations**: 60 cases ($30.0\%$).
- **Baseline 1 (Majority)** predicted `AUTO_HANDLE` on 100% of cases, resulting in **60 dangerous false auto-handles**.
- **Baseline 2 (TF-IDF)** achieved an Escalation F1 of **0.4966**, reducing dangerous false auto-handles to **24 / 60**.

---

## 4. TF-IDF Retrieval Performance & Similarity Distribution

- **Historical Retrieval Corpus Size**: 8,000 cases
- **Mean Cosine Similarity**: 0.2822
- **Median Cosine Similarity**: 0.2611
- **Queries with Similarity $\ge 0.5$**: 5 / 200 (2.5%)
- **Queries with Similarity $\ge 0.3$**: 61 / 200 (30.5%)

### Top 5 Strongest Retrieval Matches:
1. **Sim: 0.6917** | *Query*: "@AmazonHelp where is my parcel!!!"
   - *Matched*: "@AmazonHelp Where is my fone assholes"
   - *Resolution*: "[CUSTOMER] Sorry to hear that, Anurag. Please report this to our team here: [URL] and we’ll get this checked. [URL]"

2. **Sim: 0.6165** | *Query*: "Someone stole my amazon package from my porch"
   - *Matched*: "@AmazonHelp someone stole my package!!!"
   - *Resolution*: "[CUSTOMER] Oh no! We'd be happy to look into available options with you. When you have a moment, click here: [URL]"

3. **Sim: 0.5753** | *Query*: "Where is my amazon order 😡"
   - *Matched*: "Where is my uno cards [CUSTOMER]"
   - *Resolution*: "[CUSTOMER] Oh no! Have we missed the delivery date given on your confirmation email?"

4. **Sim: 0.5669** | *Query*: "Guys never trust [CUSTOMER] they have cheated with me plz see attachment how they have cheated me. They are not giving proof.[CUSTOMER] return my money. [URL]"
   - *Matched*: "[CUSTOMER] cheated with me plz return my money.. [URL]"
   - *Resolution*: "[CUSTOMER] We've responded to your query here: [URL] Kindly check.
Please don't provide your order details, we consider it to be personal information. Our page is visible to the public."

5. **Sim: 0.5236** | *Query*: "Oh how nice! My [CUSTOMER] account seems to have been hacked..."
   - *Matched*: "So my [CUSTOMER] account has been hacked. #ireallywishiwasjoking"
   - *Resolution*: "[CUSTOMER] I'm sorry for the account issues. What account are you trying to get access to (.com, .co.uk, .fr)?"

### Top 5 Weakest Retrieval Matches:
1. **Sim: 0.1781** | *Query*: "[CUSTOMER] package for my boyfriends birthday arrived today &amp; the contents were broken !!!!???"
   - *Matched*: "Hey [CUSTOMER] thanks for leaving my package from [CUSTOMER] out in the pouring rain today. Thankfully the contents seem ok, but what the heck? Glad I was checking periodically or it could have been ruined."
   - *Resolution*: "[CUSTOMER] I'm sorry your package got wet! Let us help you file the proper carrier feedback here: [URL]"

2. **Sim: 0.1772** | *Query*: "[CUSTOMER] your listing showed the item on the left and promised “no wires” what I got was on the right and they want to deduct postage from the refund. Very unimpressed. [URL]"
   - *Matched*: "@AmazonHelp are we having issues finding the right place? We live five minutes from the LEEDS depot 😂😂😂😂 [URL]"
   - *Resolution*: "[CUSTOMER] Oh my! I'm sorry about that. We would like to look into this for you in real time. Please reach out to us when you have a moment: [URL]"

3. **Sim: 0.1758** | *Query*: "@AmazonHelp U dont missed time frame but prime delivery to a location within 24hrs
But for me it takes 4days to reach that location.
Plus it came with same courier. They dont deliver to my location. They are giving to some other courier. Pathetic delivery."
   - *Matched*: "[CUSTOMER] Your customer representative is not behaving properly. They are giving fake information always."
   - *Resolution*: "[CUSTOMER] We apologize for the hassle caused. Kindly let us know where did we go wrong, we would like to work on it."

4. **Sim: 0.1740** | *Query*: "I have no words right now, describing how I am feeling, [CUSTOMER] You failed to provide the product on day one and then tell me it's delayed and you ruined my day, chances are I will not pre-order games from you guys. [URL]"
   - *Matched*: "[CUSTOMER] You guys are fake one day delivery sucks.... [URL]"
   - *Resolution*: "[CUSTOMER] I'm sorry about the unpleasant experience. Kindly contact us here: [URL] and we'll sort this for you."

5. **Sim: 0.1607** | *Query*: "I never know which UPS, FedEx, or Amazon driver I’ll get. Some of them know the building code &amp; some don’t. It could delay me getting my stuff by a day or more."
   - *Matched*: "@AmazonHelp Hi Amazon, could I get some assistance please?"
   - *Resolution*: "[CUSTOMER] We'd be happy to help any way we can! Without providing any account details, can you let us know what's going on?"

