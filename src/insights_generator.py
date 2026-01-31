import pandas as pd

def generate_insights(df):
    insights = []

    # -----------------------------
    # Average Spending
    # -----------------------------
    if "Purchase Amount (USD)" in df.columns:
        avg_spend = pd.to_numeric(
            df["Purchase Amount (USD)"], errors="coerce"
        ).mean()
        insights.append(f"💰 Average customer spending is ${avg_spend:.2f}")

    # -----------------------------
    # Largest Cluster
    # -----------------------------
    if "Cluster" in df.columns:
        top_cluster = df["Cluster"].value_counts().idxmax()
        insights.append(
            f"👥 Cluster {top_cluster} contains the highest number of customers."
        )

    # -----------------------------
    # High Value Customers
    # -----------------------------
    if "Purchase Amount (USD)" in df.columns:
        high_value = df[
            pd.to_numeric(df["Purchase Amount (USD)"], errors="coerce")
            > avg_spend
        ]
        insights.append(
            f"🔥 {len(high_value)} customers spend above average and are high-value users."
        )

    # -----------------------------
    # Frequency Insight (SAFE)
    # -----------------------------
    if "Frequency of Purchases" in df.columns:
        freq_numeric = pd.to_numeric(
            df["Frequency of Purchases"], errors="coerce"
        )

        if freq_numeric.notna().sum() > 0:
            freq_avg = freq_numeric.mean()
            insights.append(
                f"🔁 Average purchase frequency is {freq_avg:.1f}."
            )
        else:
            insights.append(
                "🔁 Purchase frequency data is categorical and used for segmentation."
            )

    return insights
