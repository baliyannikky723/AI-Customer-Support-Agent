# Intent Boundary & Disambiguation Analysis Report

This report analyzes overlapping customer intent boundaries, defines operational disambiguation rules, and details why certain categories were merged or kept distinct.

---

## 1. Difficult Boundary Pairs & Operational Disambiguation Rules

### Boundary 1: `late_delivery_complaint` vs. `delivery_status_tracking`
- **The Confusion**: A customer asking *"Where is my package, it was supposed to come today?"* mentions both tracking and delay.
- **Root Disambiguation Rule**:
  - **`late_delivery_complaint`**: The promised estimated delivery date/time has **explicitly passed**, or the customer expresses frustration over a delay / missed Prime guarantee. (Operational resolution: checking delay compensation, investigating courier bottleneck).
  - **`delivery_status_tracking`**: The package is **in-transit within its expected delivery window**, or the customer simply requests tracking updates, carrier name, or dispatch confirmation. (Operational resolution: providing self-serve tracking link `[URL]`).

---

### Boundary 2: `return_and_pickup_inquiry` vs. `refund_status_and_request`
- **The Confusion**: Customers often return items specifically to get a refund (*"I want to return this and get my money back"*).
- **Root Disambiguation Rule**:
  - **`return_and_pickup_inquiry`**: Focuses on the **physical return logistics** (return policy window, label generation, carrier pickup delay, dropping package off).
  - **`refund_status_and_request`**: The item has **already been returned or cancelled**, and the customer is querying the financial refund transaction, timeline, or missing credit.

---

### Boundary 3: `order_delivered_not_received` vs. `damaged_defective_or_wrong_item`
- **The Confusion**: Missing package delivery vs. delivered package with missing/wrong items inside.
- **Root Disambiguation Rule**:
  - **`order_delivered_not_received`**: The entire package is absent (carrier marked "Delivered" or "Handed to resident", but nothing arrived at the doorstep/porch).
  - **`damaged_defective_or_wrong_item`**: A package was physically received, but the contents are broken, scratched, missing components, or an incorrect product was delivered (e.g. received stone instead of phone).

---

### Boundary 4: `payment_and_billing_issues` vs. `prime_membership_and_digital`
- **The Confusion**: An unexpected $12.99 / $99 charge on a credit card statement.
- **Root Disambiguation Rule**:
  - **`prime_membership_and_digital`**: Explicitly mentions Amazon Prime membership fees, Prime Video / Music subscription charges, or digital Kindle/Audible purchases.
  - **`payment_and_billing_issues`**: General order payment declines, double-charging for retail physical orders, gift card redemption errors, or bank checkout failures.

---

## 2. Merged vs. Rejected Categories

| Proposed Sub-Category | Decision | Rationale |
|---|---|---|
| *Courier Driver Conduct* | **Merged into `feedback_or_complaint` / `late_delivery_complaint`** | Rare as an isolated category (~0.4%); does not require a separate retrieval policy. |
| *Gift Card Balance Inquiries* | **Merged into `payment_and_billing_issues`** | Shares the same payment verification and wallet troubleshooting workflow. |
| *Kindle Hardware vs App* | **Merged into `prime_membership_and_digital`** | Both follow Amazon digital services & device troubleshooting workflows. |
| *Return Pickup Reschedule* | **Kept in `return_and_pickup_inquiry`** | Directly connected to Amazon's reverse logistics process. |

---

## 3. Multi-Intent Hierarchy (Primary Intent Rule)

When a customer message contains multiple grievances (e.g., *"My package arrived late AND it was damaged"*):
1. **Physical Product Condition Precedence**: If an item is damaged or defective, the return/replacement protocol takes priority over a delivery delay $\rightarrow$ `damaged_defective_or_wrong_item`.
2. **Financial Action Precedence**: If a customer explicitly asks for a refund after a cancelled or missing order $\rightarrow$ `refund_status_and_request`.
