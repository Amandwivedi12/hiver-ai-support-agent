import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, confusion_matrix

HUMAN_FILE = Path("data/human_scores_completed.xlsx")
LLM_FILE = Path("data/llm_judge_results.csv")

OUTPUT_METRICS = Path("data/llm_human_agreement.csv")
OUTPUT_MATRIX = Path("data/llm_human_confusion_matrix.csv")


# Load scores
human = pd.read_excel(HUMAN_FILE)[["tweet_id", "human_score"]]
llm = pd.read_csv(LLM_FILE)[["tweet_id", "llm_score"]]

# Merge using tweet_id
df = human.merge(llm, on="tweet_id")

# Keep valid scores only
df = df[
    df["human_score"].isin([0, 1, 2, 3])
    & df["llm_score"].isin([0, 1, 2, 3])
].copy()

# Metrics
exact_agreement = (df["human_score"] == df["llm_score"]).mean()
adjacent_agreement = (
    (df["human_score"] - df["llm_score"]).abs() <= 1
).mean()

quadratic_kappa = cohen_kappa_score(
    df["human_score"],
    df["llm_score"],
    weights="quadratic"
)

spearman = df["human_score"].corr(
    df["llm_score"],
    method="spearman"
)

mean_human = df["human_score"].mean()
mean_llm = df["llm_score"].mean()

# Save metrics
metrics = pd.DataFrame([
    ["samples", len(df)],
    ["exact_agreement", exact_agreement],
    ["within_1_agreement", adjacent_agreement],
    ["quadratic_weighted_kappa", quadratic_kappa],
    ["spearman_correlation", spearman],
    ["mean_human_score", mean_human],
    ["mean_llm_score", mean_llm],
    ["mean_score_gap_llm_minus_human", mean_llm - mean_human],
], columns=["metric", "value"])

metrics.to_csv(OUTPUT_METRICS, index=False)

# Confusion matrix
matrix = confusion_matrix(
    df["human_score"],
    df["llm_score"],
    labels=[0, 1, 2, 3]
)

matrix_df = pd.DataFrame(
    matrix,
    index=["Human_0", "Human_1", "Human_2", "Human_3"],
    columns=["LLM_0", "LLM_1", "LLM_2", "LLM_3"]
)

matrix_df.to_csv(OUTPUT_MATRIX)

print()
print("=== LLM vs HUMAN AGREEMENT ===")
print(f"Samples: {len(df)}")
print(f"Exact agreement: {exact_agreement:.2%}")
print(f"Within ±1 agreement: {adjacent_agreement:.2%}")
print(f"Quadratic weighted kappa: {quadratic_kappa:.4f}")
print(f"Spearman correlation: {spearman:.4f}")
print(f"Mean human score: {mean_human:.2f}/3")
print(f"Mean LLM score: {mean_llm:.2f}/3")
print(f"LLM - human gap: {mean_llm - mean_human:+.2f}")

print()
print("Confusion matrix:")
print(matrix_df)

print()
print(f"Saved: {OUTPUT_METRICS}")
print(f"Saved: {OUTPUT_MATRIX}")