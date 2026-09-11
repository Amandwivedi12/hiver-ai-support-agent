import pandas as pd
from pathlib import Path

INPUT_FILE = Path("twcs/twcs.csv")
OUTPUT_FILE = Path("data/sprintcare_customer_messages.csv")

print("Loading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ]
)

# Normalize tweet IDs so numeric/string formats match correctly
df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce").astype("Int64")
df["response_tweet_id"] = pd.to_numeric(
    df["response_tweet_id"], errors="coerce"
).astype("Int64")
df["in_response_to_tweet_id"] = pd.to_numeric(
    df["in_response_to_tweet_id"], errors="coerce"
).astype("Int64")

# SprintCare official tweets
brand_tweets = df[df["author_id"].astype(str).str.lower() == "sprintcare"]

brand_ids = set(brand_tweets["tweet_id"].dropna())

print(f"SprintCare tweets found: {len(brand_ids):,}")

# Customer messages directly replying to SprintCare
customers = df[
    (df["inbound"].astype(str).str.lower() == "true") &
    (df["in_response_to_tweet_id"].isin(brand_ids))
].copy()

customers = customers[
    [
        "tweet_id",
        "created_at",
        "text",
        "in_response_to_tweet_id"
    ]
]

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
customers.to_csv(OUTPUT_FILE, index=False)

print()
print("Done!")
print(f"SprintCare customer messages: {len(customers):,}")
print(f"Saved to: {OUTPUT_FILE}")