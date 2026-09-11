\# Decision Log



\## 1. Selected SprintCare as the target brand



\*\*Decision:\*\* Use SprintCare as the single brand for the support-agent prototype.



\*\*Why:\*\* SprintCare had a sufficiently large number of customer-support interactions while keeping the project scope manageable. Focusing on one brand also makes historical-response retrieval more consistent.



\---



\## 2. Used direct customer-to-brand replies



\*\*Decision:\*\* Define the working customer-support dataset as inbound customer tweets whose `in\_response\_to\_tweet\_id` points directly to a SprintCare tweet.



\*\*Why:\*\* This creates a clean relationship between the customer's request and the brand's historical response, which is necessary for grounded retrieval.



\---



\## 3. Reduced the problem to a compact intent taxonomy



\*\*Decision:\*\* Use 13 support intents instead of attempting to reproduce every possible topic in the raw Twitter dataset.



\*\*Why:\*\* A compact taxonomy is easier to evaluate consistently and better suited to a customer-support routing prototype.



\---



\## 4. Added an OTHER intent



\*\*Decision:\*\* Include `OTHER` for messages that do not provide enough evidence for one of the defined support categories.



\*\*Why:\*\* Forcing every ambiguous social-media message into a specific intent would create artificial classification confidence.



\---



\## 5. Created a 200-example golden set



\*\*Decision:\*\* Use 200 sampled customer messages for the primary classification evaluation.



\*\*Why:\*\* The assignment prioritizes proof over running the complete dataset. A manually reviewed sample provides a practical evaluation set while keeping review effort manageable.



Sampling uses:



```python

sample(200, random\_state=2026)

```



\---



\## 6. Used manual review for golden labels



\*\*Decision:\*\* The final intent labels in the golden set were manually reviewed.



\*\*Why:\*\* Automated labels would not provide a reliable ground truth for measuring classifier performance. Human review was used to resolve ambiguous examples.



\---



\## 7. Included a majority-class baseline



\*\*Decision:\*\* Use the majority class as the trivial baseline.



\*\*Why:\*\* A non-trivial classifier should demonstrate improvement over a classifier that always predicts the most frequent intent.



\---



\## 8. Included a keyword classifier baseline



\*\*Decision:\*\* Use a deterministic keyword-based classifier as the simple baseline.



\*\*Why:\*\* It is transparent, fast, easy to reproduce, and provides a meaningful comparison point before introducing retrieval-based support behavior.



\---



\## 9. Used TF-IDF for historical retrieval



\*\*Decision:\*\* Use TF-IDF with unigram and bigram features for retrieving similar historical customer messages.



\*\*Why:\*\* TF-IDF is lightweight, interpretable, requires no external API, and is sufficient to establish a reproducible retrieval baseline.



\---



\## 10. Excluded self-matches during retrieval



\*\*Decision:\*\* The retrieval system excludes the current customer's own tweet from the candidate set.



\*\*Why:\*\* Without this safeguard, a message could retrieve itself and produce an artificially high similarity score, making the evaluation misleading.



\---



\## 11. Used the strongest retrieved case as the primary grounding source



\*\*Decision:\*\* Retrieve multiple candidates but use the highest-scoring historical case as the primary grounding example.



\*\*Why:\*\* This keeps the prototype simple and makes the source of each generated response easy to inspect.



The output still records the retrieved customer message, historical reply, and similarity score for analysis.



\---



\## 12. Added similarity thresholds



\*\*Decision:\*\* Use three retrieval-confidence bands:



\* High: `>= 0.50`

\* Medium: `0.30–0.50`

\* Low: `< 0.30`



\*\*Why:\*\* A single similarity threshold does not distinguish strong matches from weak ones. The three bands allow the system to behave more conservatively when historical evidence is weak.



\---



\## 13. Added explicit escalation rules



\*\*Decision:\*\* Escalate security-sensitive, urgent, legal, strongly frustrated, account-related, and unknown requests.



\*\*Why:\*\* A support agent should not attempt to automatically resolve every message. High-risk or ambiguous requests should be routed to a human rather than relying on weak retrieval evidence.



\---



\## 14. Evaluated a 500-message agent demonstration



\*\*Decision:\*\* Run the complete support-agent workflow on a reproducible 500-message sample rather than the full customer dataset.



Sampling uses:



```python

sample(500, random\_state=2026)

```



\*\*Why:\*\* The assignment explicitly encourages subsampling. A fixed sample makes the headline results reproducible while avoiding unnecessary full-dataset computation.



\---



\## 15. Added a separate human reply-quality review



\*\*Decision:\*\* Manually review a random sample of 50 generated replies.



Sampling uses:



```python

sample(50, random\_state=99)

```



\*\*Why:\*\* Automated lexical checks can overestimate response quality. Human review provides qualitative evidence about whether retrieved responses are actually relevant, useful, and appropriately grounded.



The completed review is stored in:



```text

data/human\_scores\_completed.xlsx

```



The human review was particularly useful for identifying retrieval mismatches, generic fallbacks, sensitive-issue handling problems, and language/tone mismatches.



