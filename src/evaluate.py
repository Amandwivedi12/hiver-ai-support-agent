import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

GOLDEN_FILE = Path("data/golden_set_reviewed.csv")
AGENT_FILE = Path("data/sprintcare_agent_demo.csv")
OUTPUT_FILE = Path("data/evaluation_results.csv")


# Same keyword rules used by the support agent
INTENT_KEYWORDS = {
    "NETWORK_COVERAGE": [
        "coverage", "signal", "tower", "lte", "4g", "service",
        "no service", "spotty", "connection", "area", "roaming"
    ],
    "MOBILE_DATA_ISSUE": [
        "data", "internet", "online", "mobile data", "wifi",
        "can't get online", "data not working"
    ],
    "CALLING_ISSUE": [
        "call", "calling", "calls", "text", "texts", "sms",
        "dropped call", "can't call", "cannot call"
    ],
    "BILLING_CHARGES": [
        "bill", "billing", "charge", "charged", "payment",
        "pay", "money", "credit", "refund", "cost", "collections"
    ],
    "PLAN_OR_PRICING": [
        "plan", "plans", "price", "pricing", "deal", "discount",
        "promotion", "promo", "offer", "free", "upgrade price"
    ],
    "DEVICE_OR_UPGRADE": [
        "phone", "iphone", "android", "device", "upgrade",
        "screen", "battery", "warranty", "sim", "handset"
    ],
    "ACCOUNT_OR_SECURITY": [
        "account", "password", "login", "logged", "security",
        "ssn", "social security", "bank account", "locked"
    ],
    "SHIPPING_OR_ORDER": [
        "ship", "shipping", "delivery", "delivered", "order",
        "preorder", "pre-order", "box", "return", "ups"
    ],
    "CUSTOMER_SERVICE": [
        "customer service", "agent", "support", "help",
        "call back", "callback", "representative", "department"
    ],
    "DM_OR_FOLLOW_UP": [
        "dm", "direct message", "private message", "message",
        "messaged", "sent you a message", "check your dm"
    ],
    "WEBSITE_OR_APP": [
        "website", "site", "app", "browser", "page",
        "link", "load", "error message", "online chat"
    ],
    "GENERAL_FEEDBACK": [
        "horrible", "terrible", "sucks", "leaving sprint",
        "switching", "bad service", "disappointed", "rude"
    ],
}


def classify(text):
    text = str(text).lower()

    scores = {}

    for intent, keywords in INTENT_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in text:
                score += 2 if " " in keyword else 1
        scores[intent] = score

    best = max(scores, key=scores.get)

    if scores[best] == 0:
        return "OTHER"

    return best


def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print()
    print("=" * 55)
    print(model_name)
    print("=" * 55)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


print("Loading golden set...")

gold = pd.read_csv(GOLDEN_FILE)

# Remove rows without a label
gold = gold.dropna(subset=["intent"]).copy()

gold["intent"] = gold["intent"].astype(str).str.strip()

print(f"Golden examples: {len(gold)}")


# ---------------------------------------------------------
# BASELINE 1: Majority-class baseline
# ---------------------------------------------------------

majority_intent = gold["intent"].value_counts().idxmax()

majority_predictions = [
    majority_intent
] * len(gold)

results = []

results.append(
    evaluate_model(
        gold["intent"],
        majority_predictions,
        "Baseline 1 - Majority Class"
    )
)


# ---------------------------------------------------------
# BASELINE 2: Keyword classifier
# ---------------------------------------------------------

keyword_predictions = [
    classify(text)
    for text in gold["text"]
]

results.append(
    evaluate_model(
        gold["intent"],
        keyword_predictions,
        "Baseline 2 - Keyword Classifier"
    )
)


# ---------------------------------------------------------
# Detailed classification report
# ---------------------------------------------------------

print()
print("=" * 55)
print("KEYWORD CLASSIFIER - CLASSIFICATION REPORT")
print("=" * 55)

print(
    classification_report(
        gold["intent"],
        keyword_predictions,
        zero_division=0
    )
)


# ---------------------------------------------------------
# Save predictions for error analysis
# ---------------------------------------------------------

gold["predicted_intent"] = keyword_predictions
gold["correct"] = (
    gold["intent"] == gold["predicted_intent"]
)

gold.to_csv(OUTPUT_FILE, index=False)

print()
print(f"Saved detailed evaluation to: {OUTPUT_FILE}")


# ---------------------------------------------------------
# Error analysis
# ---------------------------------------------------------

errors = gold[gold["correct"] == False].copy()

print()
print("=" * 55)
print("ERROR ANALYSIS")
print("=" * 55)

print(f"Total examples : {len(gold)}")
print(f"Correct        : {gold['correct'].sum()}")
print(f"Incorrect      : {len(errors)}")

if len(gold) > 0:
    print(
        f"Accuracy       : "
        f"{gold['correct'].mean():.4f}"
    )

print()
print("Top confusion pairs:")

if len(errors) > 0:
    confusion = (
        errors
        .groupby(["intent", "predicted_intent"])
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    for (actual, predicted), count in confusion.items():
        print(
            f"  Actual: {actual:25s} "
            f"Predicted: {predicted:25s} "
            f"Count: {count}"
        )


# ---------------------------------------------------------
# Save summary
# ---------------------------------------------------------

summary = pd.DataFrame(results)

SUMMARY_FILE = Path("data/evaluation_summary.csv")
summary.to_csv(SUMMARY_FILE, index=False)

print()
print(f"Saved summary to: {SUMMARY_FILE}")

print()
print("EVALUATION COMPLETE")