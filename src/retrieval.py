import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CUSTOMER_FILE = Path("data/sprintcare_customer_messages.csv")
TWCS_FILE = Path("twcs/twcs.csv")
OUTPUT_FILE = Path("data/retrieval_results.csv")

SAMPLE_SIZE = 100


print("Loading SprintCare customer messages...")

customers = pd.read_csv(CUSTOMER_FILE)

print(f"Customer messages: {len(customers):,}")


print("Loading historical SprintCare replies...")

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
    customers["in_response_to_tweet_id"]
    .map(reply_map)
)


retrieval_data = customers.dropna(
    subset=["text", "historical_reply"]
).copy()


retrieval_data = retrieval_data.reset_index(drop=True)


print(
    f"Historical cases available: "
    f"{len(retrieval_data):,}"
)


print("Building TF-IDF retrieval index...")


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
    top_k=3
):

    query_vector = vectorizer.transform(
        [str(query)]
    )

    similarities = cosine_similarity(
        query_vector,
        customer_matrix
    ).flatten()


    if exclude_tweet_id is not None:

        matching_rows = retrieval_data.index[
            retrieval_data["tweet_id"]
            == exclude_tweet_id
        ]

        for index in matching_rows:
            similarities[index] = -1


    top_indices = similarities.argsort()[-top_k:][::-1]


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
                    float(similarities[index]),
                    4
                )
        })


    return results


print()
print("=" * 60)
print("RUNNING RETRIEVAL EVALUATION")
print("=" * 60)


sample = customers.sample(
    min(SAMPLE_SIZE, len(customers)),
    random_state=2026
).copy()


results = []


for _, row in sample.iterrows():

    retrieved = retrieve_similar(
        row["text"],
        exclude_tweet_id=row["tweet_id"],
        top_k=3
    )


    best_case = retrieved[0]


    results.append({

        "tweet_id":
            row["tweet_id"],

        "customer_message":
            row["text"],

        "similar_customer_message":
            best_case["similar_customer_message"],

        "historical_reply":
            best_case["historical_reply"],

        "retrieval_similarity":
            best_case["similarity"]
    })


result_df = pd.DataFrame(results)


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


avg_similarity = result_df[
    "retrieval_similarity"
].mean()


high_similarity = (
    result_df["retrieval_similarity"] >= 0.50
).mean()


medium_similarity = (
    (result_df["retrieval_similarity"] >= 0.30)
    &
    (result_df["retrieval_similarity"] < 0.50)
).mean()


low_similarity = (
    result_df["retrieval_similarity"] < 0.30
).mean()


print()
print("=" * 60)
print("RETRIEVAL EVALUATION COMPLETE")
print("=" * 60)


print(
    f"Queries evaluated: "
    f"{len(result_df):,}"
)


print(
    f"Average retrieval similarity: "
    f"{avg_similarity:.4f}"
)


print(
    f"High similarity (>= 0.50): "
    f"{high_similarity:.1%}"
)


print(
    f"Medium similarity (0.30–0.50): "
    f"{medium_similarity:.1%}"
)


print(
    f"Low similarity (< 0.30): "
    f"{low_similarity:.1%}"
)


print()
print(
    f"Saved to: "
    f"{OUTPUT_FILE}"
)