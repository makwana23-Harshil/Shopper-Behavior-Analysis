import pandas as pd

def generate_insights(df):
    insights = []

    # Average Spending
    avg_spend = pd.to_numeric(
        df["Purchase Amount (USD)"], errors="coerce"
    ).mean()

    insights.append(
        f"💰 Customers spend an average of ${avg_spend:.2f}, indicating moderate purchasing power."
    )

    # Dominant Cluster
    top_cluster = df["Cluster"].value_counts().idxmax()
    insights.append(
        f"👥 Cluster {top_cluster} represents the largest customer group, making it the primary target segment."
    )

    # High Value Customers
    high_value = df[
        pd.to_numeric(df["Purchase Amount (USD)"], errors="coerce") > avg_spend
    ]

    insights.append(
        f"🔥 {len(high_value)} customers spend above average and are ideal for premium offers or loyalty programs."
    )

    # Discount Sensitivity
    if "Discount Applied" in df.columns:
        discount_users = df[df["Discount Applied"] == 1]
        insights.append(
            f"🏷️ {len(discount_users)} customers respond positively to discounts, indicating price sensitivity."
        )

    # Purchase Frequency Insight
    if "Frequency of Purchases" in df.columns:
        freq = pd.to_numeric(df["Frequency of Purchases"], errors="coerce")
        if freq.notna().sum() > 0:
            insights.append(
                f"🔁 Average purchase frequency is {freq.mean():.1f}, showing repeat purchase behavior."
            )

    return insights
