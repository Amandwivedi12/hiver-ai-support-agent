import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/sprintcare_agent_demo.csv")
OUTPUT_FILE = Path("data/reply_evaluation.csv")


def score_reply(row):
    intent = row["predicted_intent"]
    reply = str(row["draft_reply"]).lower()

    score = 0
    reasons = []

    # 1. Reply should not be empty
    if reply.strip():
        score += 1
    else:
        reasons.append("Empty reply")

    # 2. Should acknowledge/help the customer
    helpful_words = [
        "help", "sorry", "assist", "check", "look",
        "review", "support", "reach"
    ]

    if any(word in reply for word in helpful_words):
        score += 1
    else:
        reasons.append("Weak acknowledgement/help")

    # 3. Should request a useful next step for actionable intents
    actionable = {
        "NETWORK_COVERAGE",
        "MOBILE_DATA_ISSUE",
        "CALLING_ISSUE",
        "BILLING_CHARGES",
        "PLAN_OR_PRICING",
        "DEVICE_OR_UPGRADE",
        "ACCOUNT_OR_SECURITY",
        "SHIPPING_OR_ORDER",
        "CUSTOMER_SERVICE",
        "WEBSITE_OR_APP"
    }

    if intent in actionable:
        if "dm" in reply or "details" in reply or "information" in reply:
            score += 1
        else:
            reasons.append("No clear next step")

    # 4. Sensitive/account issues should mention security
    if intent == "ACCOUNT_OR_SECURITY":
        if "security" in reply or "sensitive" in reply or "privately" in reply:
            score += 1
        else:
            reasons.append("Missing security guidance")

    # 5. Unknown intent should ask for more information
    if intent == "OTHER":
        if "details" in reply or "more" in reply:
            score += 1
        else:
            reasons.append("Does not request clarification")

    return pd.Series({
        "reply_score": score,
        "max_score": 4,
        "reply_quality": (
            "GOOD" if score >= 3
            else "ACCEPTABLE" if score == 2
            else "WEAK"
        ),
        "quality_reason": (
            "; ".join(reasons)
            if reasons else "Meets basic support-reply criteria"
        )
    })


print("Loading agent results...")

df = pd.read_csv(INPUT_FILE)

print(f"Agent responses: {len(df):,}")

scores = df.apply(score_reply, axis=1)

result = pd.concat([df, scores], axis=1)

result.to_csv(OUTPUT_FILE, index=False)

print()
print("=" * 55)
print("REPLY EVALUATION")
print("=" * 55)

print(
    result["reply_quality"]
    .value_counts()
)

print()
print(
    f"Average reply score: "
    f"{result['reply_score'].mean():.2f}/4"
)

print(
    f"Good replies: "
    f"{(result['reply_quality'] == 'GOOD').mean():.1%}"
)

print(
    f"Acceptable replies: "
    f"{(result['reply_quality'] == 'ACCEPTABLE').mean():.1%}"
)

print(
    f"Weak replies: "
    f"{(result['reply_quality'] == 'WEAK').mean():.1%}"
)

print()
print(f"Saved to: {OUTPUT_FILE}")
print()
print("REPLY EVALUATION COMPLETE")