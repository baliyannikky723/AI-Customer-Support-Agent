# Customer Support Handling & Escalation Policy: AmazonHelp

This document defines the formal ground-truth triage policy for determining whether an incoming customer inquiry is eligible for automated resolution (`AUTO_HANDLE`) or must be transferred to a human specialist (`ESCALATE`), along with mandatory standardized escalation reasons.

---

## 1. Core Principles of Triage & Escalation

An AI Support Agent must operate with a **safety-first, trust-maximizing triage policy**:
- **`AUTO_HANDLE`**: The issue is solvable through standard self-service workflows, clear policy guidance, standard timeline clarifications, or direct self-service portal links (`[URL]`) without requiring manual exception overrides or sensitive account actions.
- **`ESCALATE`**: The issue involves account security risks, financial disputes, severe delivery failures exceeding standard carrier buffers, physical loss/theft, repeated failed courier attempts, explicit customer anger/supervisor demands, or complex multi-issue claims that require human authority.

---

## 2. Intent-by-Intent Handling Policy Matrix

| Intent Category | Eligible for `AUTO_HANDLE` | Mandatory `ESCALATE` Condition | Standard Escalation Reason |
|---|---|---|---|
| **`delivery_status_tracking`** | Normal in-transit tracking queries within expected window. | Tracking shows carrier exception, lost scan, or unresolvable tracking error. | `"Carrier tracking exception requires manual investigation"` |
| **`late_delivery_complaint`** | Delay is $<24$ hours past estimate; standard carrier buffer applies. | Order is $>48$ hours overdue past Prime guarantee; time-sensitive urgent order. | `"Package overdue past guaranteed delivery window by >48 hours"` |
| **`order_delivered_not_received`** | Initial notification $<24$ hours ago; guiding neighbor/property search. | Package still missing after property search; high-value missing item or stolen delivery. | `"High-risk missing delivery / suspected porch theft requiring carrier trace"` |
| **`damaged_defective_or_wrong_item`**| Standard item damage; directing to automated return/replacement portal. | Suspected fraudulent delivery (e.g. empty box, soap/stone inside); expensive product damage. | `"Severe defect / wrong item delivered requiring manual replacement approval"` |
| **`return_and_pickup_inquiry`** | Standard 30-day return policy; generating standard return label. | Courier failed return pickup $\ge 2$ consecutive times; expired return window exception requested. | `"Repeated courier pickup failure requiring dispatch escalation"` |
| **`refund_status_and_request`** | Clarifying standard 3–5 business day banking processing timeline. | Refund delayed $>14$ days; incorrect refund amount deducted; missing promotional credit. | `"Refund processing delay or amount discrepancy requiring financial review"` |
| **`order_cancellation_request`** | Guiding cancellation in 'Your Orders' before dispatch. | Item stuck in dispatch hold; system glitch preventing customer self-cancellation. | `"Technical dispatch lock preventing automated order cancellation"` |
| **`payment_and_billing_issues`** | Explaining temporary bank authorization hold vs. settled charge. | Unrecognized charge; double charge verified; gift card balance lost/stolen. | `"Billing discrepancy / disputed charge requiring payment team investigation"` |
| **`prime_membership_and_digital`** | Managing Prime auto-renewal; basic Prime Video / Echo app setup. | Unauthorized Prime subscription charge; persistent digital service entitlement error. | `"Prime billing dispute / digital entitlement failure requiring account review"` |
| **`account_access_and_security`** | **NEVER AUTO-HANDLED** (High Security Risk). | **ALL** login failures, OTP issues, locked accounts, and suspected account compromises. | `"Security policy: Account access and security issues mandate human specialist verification"` |
| **`other_unknown`** | General polite inquiries; asking clarifying questions. | Explicit demand to speak to human supervisor; legal threat or severe grievance. | `"Unresolved complex customer grievance requiring human supervisor"` |

---

## 3. General Escalation Triggers Across All Intents

Regardless of the classified intent, the message **MUST BE ESCALATED** if any of the following triggers are present:
1. **Explicit Human Agent Request**: *"I want to talk to a real person / supervisor / human."*
2. **Account Security Risk**: Mention of unauthorized account access, hacked password, or unexpected password reset emails.
3. **Severe Multi-Turn Frustration**: Repeated failures over multiple days ($>3$ turns) where self-service links have already failed.
4. **Legal / Regulatory / Severe Loss Threats**: Customer mentions filing a consumer court complaint, police report, or bank fraud dispute.
