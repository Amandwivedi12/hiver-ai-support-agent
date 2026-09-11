# AI Customer Support Agent — SprintCare

## Overview

This project builds a lightweight AI customer-support system using historical Twitter customer-support conversations from the `thoughtvector/customer-support-on-twitter` dataset.

The system focuses on one brand, **SprintCare**, and performs three tasks:

1. Classifies incoming customer messages into support intents.
2. Retrieves historically similar SprintCare customer cases and their responses.
3. Decides whether the request can be auto-handled or should be escalated to a human.

The implementation is intentionally lightweight and reproducible using Python, pandas, scikit-learn, and TF-IDF retrieval.

The main goal is not to claim production-level performance, but to build a reproducible prototype and provide evidence about where the approach works and where it fails.

---

## Dataset

Dataset: `thoughtvector/customer-support-on-twitter`

The dataset contains Twitter customer-support conversations with inbound customer messages and outbound brand responses.

Selected brand: **SprintCare**

From the dataset:

- SprintCare tweets: **22,381**
- Customer messages directly replying to SprintCare: **8,894**
- Golden evaluation set: **200 manually reviewed examples**
- Agent demonstration run: **500 customer messages**

The full dataset is kept locally in:

    twcs/twcs.csv

The full dataset is intentionally not committed to this repository.

---

## Project Structure

    hiver-assignment/
    ├── data/
    │   ├── sprintcare_customer_messages.csv
    │   ├── sprintcare_agent_demo.csv
    │   ├── golden_set_reviewed.csv
    │   ├── human_scores_completed.xlsx
    │   ├── llm_judge_sample.csv
    │   ├── llm_judge_results.csv
    │   ├── llm_human_agreement.csv
    │   ├── llm_human_confusion_matrix.csv
    │   ├── evaluation_results.csv
    │   ├── evaluation_summary.csv
    │   ├── retrieval_results.csv
    │   └── reply_evaluation.csv
    │
    ├── src/
    │   ├── prepare_data.py
    │   ├── support_agent.py
    │   ├── retrieval.py
    │   ├── evaluate.py
    │   ├── evaluate_replies.py
    │   ├── llm_judge.py
    │   └── llm_human_agreement.py
    │
    ├── DECISIONS.md
    ├── README.md
    ├── requirements.txt
    └── .gitignore

---

## Requirements

Python 3.10+ is recommended.

Install dependencies:

    pip install -r requirements.txt

The main dependencies are:

    pandas==3.0.5
    scikit-learn==1.9.1
    openpyxl==3.1.5
    transformers==5.17.0
    torch==2.14.0

The LLM-as-Judge uses a small local Hugging Face model and does not require an external API key.

---

## Reproduce the Pipeline

Run the following commands from the project root.

### 1. Prepare SprintCare customer data

    python src/prepare_data.py

This extracts customer messages that directly replied to SprintCare tweets and saves:

    data/sprintcare_customer_messages.csv

Expected result:

    SprintCare tweets found: 22,381
    SprintCare customer messages: 8,894

---

### 2. Run the support agent

    python src/support_agent.py

The agent follows this pipeline:

    Customer message
           ↓
    Intent classification
           ↓
    TF-IDF historical retrieval
           ↓
    Best historical customer/reply case
           ↓
    Grounded draft reply
           ↓
    Auto-handle / Escalate

The retrieval component uses TF-IDF with unigram and bigram features and cosine similarity.

Self-matches are explicitly excluded during retrieval.

The agent evaluates a reproducible 500-message sample using:

    random_state=2026

Output:

    data/sprintcare_agent_demo.csv

Headline results from the 500-message run:

- Average retrieval similarity: **0.5051**
- High similarity (≥ 0.50): **38.6%**
- Low similarity (< 0.30): **11.2%**
- Auto-handle: **263**
- Escalate: **237**

These numbers describe the prototype's behavior on the selected sample and should not be interpreted as production-level performance.

---

## Intent Taxonomy

The SprintCare customer messages were grouped into 13 compact intents based on recurring support themes observed in the dataset:

- `NETWORK_COVERAGE` — Coverage, signal, reception, towers, or service availability issues.
- `MOBILE_DATA_ISSUE` — Mobile data, internet connectivity, or data not working.
- `CALLING_ISSUE` — Problems making or receiving calls.
- `BILLING_CHARGES` — Bills, unexpected charges, payments, fees, or billing disputes.
- `PLAN_OR_PRICING` — Questions or complaints about plans, pricing, offers, or plan changes.
- `DEVICE_OR_UPGRADE` — Phone/device issues, upgrades, compatibility, or device-related requests.
- `ACCOUNT_OR_SECURITY` — Account access, personal information, security, fraud, or sensitive account issues.
- `SHIPPING_OR_ORDER` — Device orders, deliveries, shipping, tracking, or order status.
- `CUSTOMER_SERVICE` — General complaints, dissatisfaction, requests for support, or cases where the main issue is service experience.
- `DM_OR_FOLLOW_UP` — Requests to continue the conversation through DM or follow up on an existing case.
- `WEBSITE_OR_APP` — Website, app, login page, or online-service issues.
- `GENERAL_FEEDBACK` — General praise, criticism, comments, or feedback without a specific support issue.
- `OTHER` — Messages that do not fit the above categories or are too ambiguous to classify reliably.

The taxonomy was intentionally kept compact rather than creating a separate intent for every possible issue. This makes the classification task more practical while preserving the major support themes in the SprintCare data.

---

## Escalation Policy

The agent decides between `AUTO_HANDLE` and `ESCALATE`.

A message is escalated when it contains strong risk or frustration signals, including:

- Security or sensitive-account concerns such as fraud, hacking, stolen information, SSN, or bank-account references.
- Legal or serious-risk language such as lawsuit, illegal, collections, or similar terms.
- Strong customer frustration such as complaints about leaving the service, switching providers, requesting a manager/supervisor, or severe dissatisfaction.
- Messages classified as `ACCOUNT_OR_SECURITY`.
- Messages classified as `OTHER`, because ambiguous cases should receive additional human review rather than confident automated handling.

Other messages can be marked `AUTO_HANDLE` when the retrieved historical context provides a sufficiently similar support case.

---

## Reply Grounding

The support agent uses TF-IDF similarity over historical SprintCare customer messages.

For each incoming message:

1. Retrieve the most similar historical customer-support cases.
2. Exclude the incoming message itself from retrieval.
3. Use the strongest historical match as the primary grounding context.
4. Use the historical SprintCare response associated with that case to draft the reply.
5. Apply similarity thresholds:
   - `>= 0.50`: strong historical match; reuse the historical resolution directly.
   - `0.30–0.50`: moderate match; use the historical response with an additional verification reminder.
   - `< 0.30`: weak match; avoid relying on the retrieved response and ask the customer for clarification.

This approach intentionally favors historical support evidence over a fully generative response, making it easier to inspect where a draft came from.

---

## Evaluation

The intent classifier was evaluated on a 200-example golden set that was sampled from SprintCare customer messages and manually reviewed.

Two baselines were included.

### Baseline 1 — Majority Class

Always predicts the most frequent intent in the golden set.

- Accuracy: `20.00%`
- Weighted Precision: `4.00%`
- Weighted Recall: `20.00%`
- Weighted F1: `6.67%`

### Baseline 2 — Keyword Classifier

Uses simple keyword rules to map customer messages to intents.

- Accuracy: `53.00%`
- Weighted Precision: `61.79%`
- Weighted Recall: `53.00%`
- Weighted F1: `49.94%`

The keyword baseline substantially outperforms the majority baseline, showing that the taxonomy contains meaningful lexical signals but also leaves significant room for improvement.

---

## Reply Evaluation

The reply evaluation checks whether generated drafts are non-empty, helpful, actionable, and appropriate for the detected issue.

On the 500-message agent demo:

- `GOOD`: 200 (`40.0%`)
- `ACCEPTABLE`: 207 (`41.4%`)
- `WEAK`: 93 (`18.6%`)
- Average heuristic score: `2.21 / 4`

A separate local LLM-as-Judge evaluation was also run using `Qwen/Qwen2.5-0.5B-Instruct`.

A sample of 50 replies was evaluated. After merging with human scores, **49 valid paired results** were available.

LLM judge scores on the sampled replies:

- Score 0: `1`
- Score 1: `2`
- Score 2: `23`
- Score 3: `23`
- Mean LLM score: `2.39 / 3`

The heuristic evaluator is treated as a lightweight automated signal, not as a replacement for human evaluation.

---

## Human Evaluation

A random sample of 50 agent replies was manually reviewed using the same 0–3 quality scale.

The human evaluator considered:

- Relevance to the customer's actual issue
- Helpfulness
- Whether a clear next step was provided
- Quality of the retrieved historical context
- Safety
- Whether the response made unsupported claims
- Whether the response appropriately handled frustration or sensitive issues

The paired human evaluation produced a mean score of approximately `1.37 / 3` over the 49 valid paired examples.

Human review is treated as the primary quality reference because several retrieved responses were clearly unrelated or overly generic.

---

## LLM-as-Judge vs Human Agreement

The local Qwen judge was compared against the human scores on 49 valid paired examples.

Results:

- Exact agreement: `26.53%`
- Agreement within ±1 point: `71.43%`
- Quadratic weighted kappa: `-0.0204`
- Spearman correlation: `-0.0581`
- Mean human score: `1.37 / 3`
- Mean LLM score: `2.39 / 3`
- LLM-human score gap: `+1.02`

The agreement is weak. The LLM judge is noticeably more generous than the human evaluator, so the human scores are treated as the primary reference for reply quality.

This comparison is included specifically to avoid presenting the LLM judge as a reliable substitute for human evaluation.

---

## Key Failure Modes

### 1. Wrong retrieval match

The biggest problem is retrieving a historically similar-looking message that actually represents a different issue.

For example, one pricing complaint retrieved a tower/coverage-related response. Another message received a response about an unrelated Apple/device case.

**Hypothesis:** TF-IDF lexical similarity is not sufficient to understand support intent.

### 2. Generic fallback replies

Low-similarity cases often receive clarification requests such as asking the customer to provide more details. These are safe but can feel repetitive and unhelpful.

**Hypothesis:** The agent needs intent-aware fallback templates rather than one generic fallback.

### 3. Customer-service vs OTHER confusion

The keyword classifier frequently predicts `OTHER` for general complaints that humans label as `CUSTOMER_SERVICE`.

**Hypothesis:** Customer-service complaints are highly varied and often lack strong domain-specific keywords.

### 4. Context and tone are missed

Some customers express frustration, sarcasm, churn risk, or dissatisfaction without using obvious escalation keywords.

**Hypothesis:** Escalation needs semantic/tone features instead of only keyword triggers.

### 5. Sensitive issues may not escalate strongly enough

Some messages mention SSN, bank information, or security concerns while the resulting reply remains too generic.

**Hypothesis:** Security-related patterns should have higher-priority deterministic escalation rules and should never depend only on retrieval similarity.

---

## What Is Misleading About My Headline Number?

The `53%` intent accuracy is useful but can be misleading if presented as the overall quality of the support agent.

It is measured on a 200-example golden set and comes from the simple keyword classifier, not from a full end-to-end evaluation of the complete AI support system.

Similarly, the `0.5051` average retrieval similarity on the 500-message demo does not mean that half of the replies are correct. Similarity is only a retrieval signal, and manual review showed that some high-looking lexical matches were still wrong.

The reply-quality results are more revealing: human review found substantially weaker performance than the automated/LLM scores suggest.

Therefore, the headline result should be interpreted as evidence that the pipeline is working and measurable, not as proof that the agent is production-ready.

---

## Reproducing Evaluation

From the repository root:

    python src/prepare_data.py

This extracts SprintCare customer messages from the original Twitter customer-support dataset.

Then run:

    python src/support_agent.py

This generates the 500-message agent demo:

    data/sprintcare_agent_demo.csv

Run intent evaluation:

    python src/evaluate.py

This generates:

    data/evaluation_results.csv
    data/evaluation_summary.csv

Run reply evaluation:

    python src/evaluate_replies.py

This generates:

    data/reply_evaluation.csv

Run the local LLM judge:

    python src/llm_judge.py

This generates:

    data/llm_judge_results.csv

Run the human-vs-LLM agreement analysis:

    python src/llm_human_agreement.py

This generates:

    data/llm_human_agreement.csv
    data/llm_human_confusion_matrix.csv

The full Twitter dataset is intentionally not committed to the repository. The pipeline works on the SprintCare subset and evaluation samples rather than requiring a full 3M-tweet end-to-end run.

---

## Limitations

- TF-IDF retrieval is lexical and does not fully understand semantic similarity.
- Historical replies can contain outdated or context-specific information.
- Reusing historical replies verbatim can produce inappropriate responses when the retrieved case is only superficially similar.
- The keyword intent classifier is intentionally simple and has substantial room for improvement.
- `OTHER` is broad and absorbs many ambiguous customer messages.
- The escalation policy is primarily rule-based.
- The local LLM judge is small and showed weak agreement with human judgments.
- The golden set contains 200 examples, so rare intents have limited evaluation support.
- The reply-quality evaluation should be expanded with more human-reviewed examples before making production claims.

---

## Next Week Plan

1. Replace TF-IDF retrieval with embedding-based semantic retrieval.
2. Add intent-aware retrieval so similar cases are searched within the predicted intent where possible.
3. Build stronger fallback responses for low-similarity cases.
4. Add explicit safety policies for sensitive information, fraud, account security, and legal issues.
5. Add multilingual and tone-aware handling.
6. Expand the golden set and stratify it by intent.
7. Run a larger human evaluation focused on retrieval correctness and response usefulness.
8. Compare multiple retrieval strategies using recall@k and human relevance judgments.
9. Improve the escalation policy using calibrated risk signals instead of keyword matching alone.
10. Add confidence calibration so auto-handle decisions have measurable precision/coverage trade-offs.

---

## Decision Log

1. **SprintCare selected as the brand** — It provided enough direct customer-support interactions for a meaningful experiment.
2. **Direct customer-to-brand replies selected** — This creates a cleaner customer-support dataset than mixing unrelated tweets.
3. **Compact 13-intent taxonomy** — Keeps classification practical while covering the major support themes.
4. **Added OTHER** — Ambiguous and out-of-scope messages need a safe fallback category.
5. **200-example golden set** — Large enough for an initial evaluation while remaining feasible to manually review.
6. **Manual golden-label review** — Labels were reviewed manually to provide a reference set for evaluation.
7. **Majority baseline** — Establishes the minimum performance level.
8. **Keyword baseline** — Provides a simple interpretable baseline before using more complex approaches.
9. **TF-IDF retrieval** — Chosen as a lightweight, transparent retrieval baseline.
10. **Self-match exclusion** — Prevents a message from retrieving itself and artificially inflating similarity.
11. **Strongest retrieved case as primary grounding** — Keeps the generated answer traceable to one historical support interaction.
12. **Similarity thresholds** — Separates strong matches, moderate matches, and weak matches.
13. **Rule-based escalation** — Provides an interpretable first safety layer for sensitive and frustrated cases.
14. **500-message agent demo** — Provides a practical end-to-end sample without requiring a full-dataset run.
15. **50-message human reply review** — Provides direct human evidence for response quality and comparison with the LLM judge.

---

## Source

Dataset:

`thoughtvector/customer-support-on-twitter`

The project uses the original Twitter customer-support dataset and focuses on SprintCare customer messages that directly respond to SprintCare support tweets.

All borrowed dataset content and methodology are used for this take-home assignment and are not presented as original data.
