import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CUSTOMER_FILE = Path("data/sprintcare_customer_messages.csv")
TWCS_FILE = Path("twcs/twcs.csv")
OUTPUT_FILE = Path("data/sprintcare_agent_demo.csv")

TOP_K = 3


# ============================================================
# INTENT TAXONOMY
# ============================================================

INTENT_KEYWORDS = {

    "NETWORK_COVERAGE": [
        "coverage", "signal", "no signal", "service area",
        "network", "bars", "reception", "tower"
    ],

    "MOBILE_DATA_ISSUE": [
        "data", "internet", "mobile data", "4g", "5g",
        "lte", "data not working", "internet not working"
    ],

    "CALLING_ISSUE": [
        "call", "calling", "phone call", "can't call",
        "cannot call", "dropped call", "calls"
    ],

    "BILLING_CHARGES": [
        "bill", "billing", "charged", "charge", "payment",
        "refund", "invoice", "fee", "fees", "overcharged"
    ],

    "PLAN_OR_PRICING": [
        "plan", "plans", "price", "pricing", "cost",
        "monthly", "unlimited", "rate"
    ],

    "DEVICE_OR_UPGRADE": [
        "phone", "device", "iphone", "android", "upgrade",
        "trade", "sim", "replacement"
    ],

    "ACCOUNT_OR_SECURITY": [
        "account", "password", "login", "locked", "security",
        "hacked", "fraud", "stolen", "verification"
    ],

    "SHIPPING_OR_ORDER": [
        "order", "shipping", "delivery", "delivered",
        "shipment", "tracking", "package"
    ],

    "CUSTOMER_SERVICE": [
        "customer service", "support", "representative",
        "agent", "help", "manager", "supervisor"
    ],

    "DM_OR_FOLLOW_UP": [
        "dm", "direct message", "message me",
        "sent you", "follow up", "inbox"
    ],

    "WEBSITE_OR_APP": [
        "website", "web", "app", "application",
        "login page", "site"
    ],

    "GENERAL_FEEDBACK": [
        "love", "hate", "great", "awesome", "terrible",
        "horrible", "disappointed", "thank you", "thanks"
    ]
}


# ============================================================
# INTENT CLASSIFICATION
# ============================================================

def classify_intent(text):

    text = str(text).lower()

    scores = {}

    for intent, keywords in INTENT_KEYWORDS.items():

        score = 0

        for keyword in keywords:

            if keyword in text:
                score += 1

        scores[intent] = score

    best_intent = max(
        scores,
        key=scores.get
    )

    if scores[best_intent] == 0:
        return "OTHER"

    return best_intent


# ============================================================
# ESCALATION
# ============================================================

URGENT_TERMS = [
    "fraud", "hacked", "stolen", "security",
    "ssn", "bank account", "collections",
    "legal", "lawsuit", "illegal"
]

FRUSTRATION_TERMS = [
    "horrible", "terrible", "sucks",
    "leaving", "switching", "complaint",
    "manager", "supervisor"
]


def escalation_decision(text, intent):

    text = str(text).lower()

    for term in URGENT_TERMS:

        if term in text:

            return (
                "ESCALATE",
                f"Sensitive or urgent issue: {term}"
            )

    for term in FRUSTRATION_TERMS:

        if term in text:

            return (
                "ESCALATE",
                f"Customer frustration: {term}"
            )

    if intent == "ACCOUNT_OR_SECURITY":

        return (
            "ESCALATE",
            "Account or security issue requires human review"
        )

    if intent == "OTHER":

        return (
            "ESCALATE",
            "Intent could not be confidently classified"
        )

    return (
        "AUTO_HANDLE",
        "Routine support request with recognizable intent"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("Loading SprintCare customer data...")

customers = pd.read_csv(
    CUSTOMER_FILE
)

print(
    f"Loaded {len(customers):,} customer messages."
)


print("Loading historical SprintCare responses...")

df = pd.read_csv(
    TWCS_FILE,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text"
    ]
)


brand_replies = df[
    df["author_id"]
    .astype(str)
    .str.lower()
    == "sprintcare"
].copy()


brand_replies["tweet_id"] = pd.to_numeric(
    brand_replies["tweet_id"],
    errors="coerce"
).astype("Int64")


customers["tweet_id"] = pd.to_numeric(
    customers["tweet_id"],
    errors="coerce"
).astype("Int64")


customers["in_response_to_tweet_id"] = pd.to_numeric(
    customers["in_response_to_tweet_id"],
    errors="coerce"
).astype("Int64")


reply_map = dict(
    zip(
        brand_replies["tweet_id"],
        brand_replies["text"]
    )
)


customers["historical_reply"] = (
    customers[
        "in_response_to_tweet_id"
    ].map(reply_map)
)


retrieval_data = customers.dropna(
    subset=[
        "text",
        "historical_reply"
    ]
).copy()


retrieval_data = retrieval_data.reset_index(
    drop=True
)


print(
    f"Historical customer/reply pairs: "
    f"{len(retrieval_data):,}"
)


# ============================================================
# TF-IDF RETRIEVAL
# ============================================================

print("Building historical retrieval index...")


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000
)


customer_matrix = vectorizer.fit_transform(
    retrieval_data["text"].astype(str)
)


print("Retrieval index ready.")


def retrieve_similar(
    query,
    exclude_tweet_id=None,
    top_k=TOP_K
):

    query_vector = vectorizer.transform(
        [str(query)]
    )

    similarities = cosine_similarity(
        query_vector,
        customer_matrix
    ).flatten()


    # Prevent self-match
    if exclude_tweet_id is not None:

        matching_rows = retrieval_data.index[
            retrieval_data["tweet_id"]
            == exclude_tweet_id
        ]

        for index in matching_rows:

            similarities[index] = -1


    top_indices = similarities.argsort()[
        -top_k:
    ][::-1]


    results = []

    for index in top_indices:

        row = retrieval_data.iloc[index]

        results.append({
            "similar_customer_message":
                row["text"],

            "historical_reply":
                row["historical_reply"],

            "similarity":
                round(
                    float(
                        similarities[index]
                    ),
                    4
                )
        })


    return results


# ============================================================
# GROUNDED REPLY
# ============================================================

def generate_grounded_reply(
    customer_message,
    intent,
    retrieved_case
):

    historical_reply = str(
        retrieved_case["historical_reply"]
    )

    similarity = retrieved_case["similarity"]


    if similarity >= 0.50:

        return historical_reply


    if similarity >= 0.30:

        return (
            historical_reply
            + " "
            + "Please verify the specific account "
              "details before sending."
        )


    return (
        "Thanks for reaching out. "
        "We'd like to understand the issue better "
        "and help with the next steps. "
        "Please send us a DM with the relevant "
        "details so our support team can assist."
    )


# ============================================================
# RUN AGENT
# ============================================================

demo = customers.sample(
    min(500, len(customers)),
    random_state=2026
).copy()


results = []


print()
print("=" * 60)
print("RUNNING AI SUPPORT AGENT")
print("=" * 60)


for _, row in demo.iterrows():

    customer_message = row["text"]


    intent = classify_intent(
        customer_message
    )


    retrieved = retrieve_similar(
        customer_message,
        exclude_tweet_id=row["tweet_id"],
        top_k=TOP_K
    )


    best_case = retrieved[0]


    decision, reason = escalation_decision(
        customer_message,
        intent
    )


    draft_reply = generate_grounded_reply(
        customer_message,
        intent,
        best_case
    )


    results.append({

        "tweet_id":
            row["tweet_id"],

        "customer_message":
            customer_message,

        "predicted_intent":
            intent,

        "similar_customer_message":
            best_case[
                "similar_customer_message"
            ],

        "historical_reply":
            best_case[
                "historical_reply"
            ],

        "retrieval_similarity":
            best_case[
                "similarity"
            ],

        "draft_reply":
            draft_reply,

        "decision":
            decision,

        "escalation_reason":
            reason
    })


result_df = pd.DataFrame(
    results
)


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("AI SUPPORT AGENT COMPLETE")
print("=" * 60)


print(
    f"Messages evaluated: "
    f"{len(result_df):,}"
)


print()
print("Intent distribution:")

print(
    result_df[
        "predicted_intent"
    ].value_counts()
)


print()
print("Handling decision:")

print(
    result_df[
        "decision"
    ].value_counts()
)


print()
print(
    f"Average retrieval similarity: "
    f"{result_df['retrieval_similarity'].mean():.4f}"
)


print(
    f"High similarity (>= 0.50): "
    f"{(result_df['retrieval_similarity'] >= 0.50).mean():.1%}"
)


print(
    f"Low similarity (< 0.30): "
    f"{(result_df['retrieval_similarity'] < 0.30).mean():.1%}"
)


print()
print(
    f"Saved to: "
    f"{OUTPUT_FILE}"
)