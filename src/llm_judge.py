import json
import pandas as pd
from pathlib import Path
from transformers import pipeline

INPUT_FILE = Path("data/llm_judge_sample.csv")
OUTPUT_FILE = Path("data/llm_judge_results.csv")

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading local LLM...")
judge = pipeline(
    "text-generation",
    model=MODEL,
    device=-1
)

print("Local LLM ready.")

RUBRIC = """
You are evaluating an AI customer-support draft reply.

Score from 0 to 3:

0 = Completely irrelevant, misleading, unsafe, or useless.
1 = Weak: generic, mostly irrelevant, or misses the main issue.
2 = Reasonable: relevant and helpful, but incomplete or somewhat generic.
3 = Strong: directly relevant, helpful, actionable, appropriately grounded,
safe, and addresses the customer's actual issue.

Consider:
- Relevance
- Helpfulness
- Clear next step
- Use of historical support context
- Safety
- Unsupported claims

Return ONLY JSON:
{"score": 0, "reason": "short explanation"}
"""


def judge_reply(row):
    prompt = f"""
{RUBRIC}

Customer message:
{row["customer_message"]}

Predicted intent:
{row["predicted_intent"]}

Retrieved similar customer message:
{row["similar_customer_message"]}

Historical support reply:
{row["historical_reply"]}

AI draft reply:
{row["draft_reply"]}
"""

    messages = [
        {
            "role": "system",
            "content": "You are a strict customer-support quality evaluator."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    output = judge(
        messages,
        max_new_tokens=100,
        do_sample=False,
        return_full_text=False
    )

    text = output[0]["generated_text"].strip()

    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        result = json.loads(text[start:end])

        score = int(result["score"])
        if score not in [0, 1, 2, 3]:
            score = -1

        reason = str(result.get("reason", ""))

    except Exception:
        score = -1
        reason = f"Parse failed: {text}"

    return score, reason


print("Loading judge sample...")
df = pd.read_csv(INPUT_FILE)

print(f"Rows to judge: {len(df)}")
print(f"Using model: {MODEL}")

scores = []
reasons = []

for i, row in df.iterrows():
    print(f"Judging {i + 1}/{len(df)}...", end="\r")

    score, reason = judge_reply(row)

    scores.append(score)
    reasons.append(reason)

df["llm_score"] = scores
df["llm_reason"] = reasons

df.to_csv(OUTPUT_FILE, index=False)

print()
print(f"Saved: {OUTPUT_FILE}")

valid = df[df["llm_score"] >= 0]

print()
print("LLM score distribution:")
print(valid["llm_score"].value_counts().sort_index())

print()
print(f"Average LLM score: {valid['llm_score'].mean():.2f}/3")