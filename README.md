\# AI Customer Support Agent — SprintCare



\## Overview



This project builds a lightweight AI customer-support system using historical Twitter customer-support conversations from the `thoughtvector/customer-support-on-twitter` dataset.



The system focuses on one brand, \*\*SprintCare\*\*, and performs three tasks:



1\. Classifies incoming customer messages into support intents.

2\. Retrieves historically similar SprintCare customer cases and their responses.

3\. Decides whether the request can be auto-handled or should be escalated to a human.



The implementation is intentionally lightweight and reproducible using Python, pandas, scikit-learn, and TF-IDF retrieval.



\---



\## Dataset



Dataset: `thoughtvector/customer-support-on-twitter`



Source: Twitter customer-support conversations containing inbound customer messages and outbound brand responses.



Selected brand: \*\*SprintCare\*\*



From the dataset:



\* SprintCare tweets: \*\*22,381\*\*

\* Customer messages directly replying to SprintCare: \*\*8,894\*\*

\* Golden evaluation set: \*\*200 manually reviewed examples\*\*

\* Agent demonstration run: \*\*500 customer messages\*\*



The full dataset is kept in:



```text

twcs/twcs.csv

```



\---



## Project Structure

```text
hiver-assignment/
├── data/
│   ├── sprintcare_customer_messages.csv
│   ├── sprintcare_agent_demo.csv
│   ├── golden_set_reviewed.csv
│   ├── golden_set_final.xlsx
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


\## Requirements



Python 3.10+ is recommended.



Install dependencies:



```bash

pip install pandas scikit-learn openpyxl transformers torch

```



\---



\## Reproduce the Pipeline



Run the following commands from the project root.



\### 1. Prepare SprintCare customer data



```bash

python src/prepare\_data.py

```



This extracts customer messages that directly replied to SprintCare tweets and saves:



```text

data/sprintcare\_customer\_messages.csv

```



Expected result:



```text

SprintCare tweets found: 22,381

SprintCare customer messages: 8,894

```



\---



\### 2. Run the support agent



```bash

python src/support\_agent.py

```



The agent:



```text

Customer message

&#x20;      ↓

Intent classification

&#x20;      ↓

TF-IDF historical retrieval

&#x20;      ↓

Best historical customer/reply case

&#x20;      ↓

Grounded draft reply

&#x20;      ↓

Auto-handle / Escalate

```



The retrieval component uses TF-IDF with unigram and bigram features and cosine similarity.



Self-matches are explicitly excluded during retrieval.



The agent evaluates a reproducible 500-message sample using:



```python

random\_state=2026

```



Output:



```text

data/sprintcare\_agent\_demo.csv

```



Headline results from the 500-message run:



\* Average retrieval similarity: \*\*0.5051\*\*

\* High similarity (≥ 0.50): \*\*38.6%\*\*

\* Low similarity (< 0.30): \*\*11.2%\*\*

\* Auto-handle: \*\*263\*\*

\* Escalate: \*\*237\*\*



\---



\## Intent Taxonomy



The system uses a compact taxonomy derived from recurring SprintCare customer-support themes:



\* `NETWORK\_COVERAGE`

\* `MOBILE\_DATA\_ISSUE`

\* `CALLING\_ISSUE`

\* `BILLING\_CHARGES`

\* `PLAN\_OR\_PRICING`

\* `DEVICE\_OR\_UPGRADE`

\* `ACCOUNT\_OR\_SECURITY`

\* `SHIPPING\_OR\_ORDER`

\* `CUSTOMER\_SERVICE`

\* `DM\_OR\_FOLLOW\_UP`

\* `WEBSITE\_OR\_APP`

\* `GENERAL\_FEEDBACK`

\* `OTHER`



The classifier is a deterministic keyword-based baseline intended to provide a transparent and reproducible starting point.



\---



\## Escalation Policy



The agent escalates:



\* Security-sensitive issues such as fraud, hacking, stolen accounts, SSN or bank-account references.

\* Legal or potentially urgent issues.

\* Strong customer-frustration signals.

\* Account/security requests.

\* Messages classified as `OTHER`.



Routine recognizable support requests are marked:



```text

AUTO\_HANDLE

```



Otherwise:



```text

ESCALATE

```



Each decision includes an explicit escalation reason in the output.



\---



\## Reply Grounding



For each incoming customer message, the system retrieves similar historical SprintCare customer cases.



The best retrieved case provides:



\* Similar historical customer message

\* Historical SprintCare response

\* Retrieval similarity score



Reply generation follows three similarity bands:



\### High similarity



For similarity ≥ 0.50, the historical SprintCare response is used as the grounded response.



\### Medium similarity



For similarity between 0.30 and 0.50, the historical response is reused with an instruction to verify account-specific details.



\### Low similarity



For similarity below 0.30, the system falls back to a clarification/DM response instead of relying strongly on the retrieved case.



\---



\## Evaluation



The evaluation uses a \*\*200-example golden set\*\*.



Two classification baselines are compared.



\### Baseline 1 — Majority Class



```text

Accuracy : 0.2000

Precision: 0.0400

Recall   : 0.2000

F1 Score : 0.0667

```



\### Baseline 2 — Keyword Classifier



```text

Accuracy : 0.5300

Precision: 0.6179

Recall   : 0.5300

F1 Score : 0.4994

```



The keyword classifier therefore substantially improves over the trivial majority-class baseline.



Run evaluation with:



```bash

python src/evaluate.py

```



Outputs:



```text

data/evaluation\_results.csv

data/evaluation\_summary.csv

```



\---



\## Reply Evaluation



A separate heuristic reply evaluator checks:



\* Whether the reply is non-empty.

\* Whether it contains basic acknowledgement/help language.

\* Whether an actionable intent receives a next step.

\* Whether account/security cases include security guidance.

\* Whether unknown intents request clarification.



Run:



```bash

python src/evaluate\_replies.py

```



Output:



```text

data/reply\_evaluation.csv

```



This heuristic evaluator is treated as an internal automated signal, not as a substitute for human evaluation.



\---



\## Human Evaluation



A random sample of \*\*50 agent replies\*\* was manually reviewed.



Sampling:



```python

sample(50, random\_state=99)

```



The completed human evaluation is stored in:



```text

data/human\_scores\_completed.xlsx

```



Each reviewed example contains a human score from \*\*0–3\*\* and a written reason.



Human score interpretation:



```text

0 = Poor / irrelevant response

1 = Weak response

2 = Reasonable but incomplete response

3 = Strong and relevant response

```



The 50 manually scored examples have an average score of:



```text

1.37 / 3

```



The human review exposed several important weaknesses that are not captured well by the automated heuristic evaluator, particularly retrieval mismatches and generic responses.



\---

\### LLM-as-Judge vs Human Agreement



A 50-example audit set was independently scored by a human reviewer and by the local Qwen2.5-0.5B-Instruct judge using the same 0–3 support-reply quality rubric.



Due to one invalid/missing LLM score, 49 examples were available for agreement analysis.



| Metric | Result |

|---|---:|

| Valid paired examples | 49 |

| Exact agreement | 26.53% |

| Agreement within ±1 score | 71.43% |

| Quadratic weighted kappa | -0.0204 |

| Spearman correlation | -0.0581 |

| Mean human score | 1.37 / 3 |

| Mean LLM score | 2.39 / 3 |

| LLM-human score gap | +1.02 |



The low agreement indicates that the local small LLM judge is substantially more generous than the human reviewer. Therefore, the LLM judge is treated as a secondary evaluation signal rather than ground truth. Human scores are used as the primary reference for this audit.



The confusion matrix is saved in:

`data/llm\_human\_confusion\_matrix.csv`



The detailed agreement metrics are saved in:

`data/llm\_human\_agreement.csv`



\## Key Failure Modes



\### 1. Retrieval mismatch



The TF-IDF retriever can select a superficially similar historical tweet whose resolution is unrelated to the actual customer problem.



Example observed during human review:



> A pricing complaint received a historical response about checking a network tower.



Hypothesis:



TF-IDF similarity is lexical and does not reliably understand intent or resolution compatibility.



\---



\### 2. Generic fallback responses



Some low-confidence cases receive a generic DM/clarification response that does not acknowledge the customer's specific issue.



Hypothesis:



The fallback is intentionally conservative but lacks a richer response-generation layer.



\---



\### 3. Customer-service vs OTHER confusion



The keyword classifier frequently maps broad customer-service requests to `OTHER`.



Observed golden-set confusion:



```text

CUSTOMER\_SERVICE → OTHER: 19

```



Hypothesis:



Messages such as "help", "are you a bot?", or broad support requests are semantically customer-service related but have weak keyword evidence.



\---



\### 4. General feedback vs OTHER



General praise, frustration, sarcasm, and short feedback messages are difficult to distinguish from ambiguous messages.



Observed confusion:



```text

GENERAL\_FEEDBACK → OTHER: 7

```



Hypothesis:



Short social-media messages often contain too little explicit intent information for keyword rules.



\---



\### 5. Sensitive issues can be under-escalated or poorly grounded



Human review found cases involving security-sensitive information where a generic response was produced despite the sensitivity of the issue.



Hypothesis:



Escalation rules depend on explicit trigger terms and should eventually be combined with a stronger safety/security classifier.



\---



\## What Is Misleading About My Headline Number?



The \*\*0.5051 average retrieval similarity\*\* should not be interpreted as "50.51% correct retrieval."



Cosine similarity is a lexical similarity measure, not a human judgement of whether the retrieved historical case is actually useful.



Human review demonstrated that some apparently similar cases were still irrelevant to the customer's real problem.



Similarly, the 53% keyword-classification accuracy is measured on a manually reviewed 200-example sample and should not be treated as performance on the entire dataset.



The results are therefore intended as reproducible evidence for the current prototype rather than claims of production-level quality.



\---



\## Reproducing Evaluation



Run:



```bash

python src/evaluate.py

```



Then:



```bash

python src/evaluate\_replies.py

```



For retrieval diagnostics:



```bash

python src/retrieval.py

```



The main generated artifacts are written to the `data/` directory.



\---



\## Limitations



1\. The intent classifier is keyword-based.

2\. TF-IDF retrieval measures lexical similarity rather than semantic resolution similarity.

3\. Retrieved historical replies can sometimes be irrelevant even when similarity is moderate/high.

4\. The reply generation layer is intentionally lightweight and does not use an external LLM API.

5\. The automated reply evaluator is heuristic and should not be considered an independent LLM judge.

6\. Human evaluation was performed on a 50-example sample rather than the complete 500-message demonstration set.

7\. The current system should be treated as a prototype rather than a production customer-support system.



\---



\## Next Week Plan



If given another week, I would prioritize:



1\. Replace keyword intent classification with a stronger few-shot or embedding-based classifier.

2\. Use semantic embeddings for historical case retrieval.

3\. Retrieve multiple candidates and rerank them using intent/resolution compatibility.

4\. Generate responses from retrieved evidence rather than copying historical replies directly.

5\. Add stronger PII/security detection and escalation rules.

6\. Expand the human evaluation set and measure agreement between human reviewers and an LLM judge.

7\. Evaluate retrieval separately using human relevance labels.

8\. Add confidence calibration for auto-handle vs escalation decisions.



\---



\## Source



Dataset:



`thoughtvector/customer-support-on-twitter`



This project uses the dataset for research and evaluation purposes. The implementation and evaluation artifacts in this repository are original work for the Hiver take-home assignment.



