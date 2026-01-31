def generate_insights(df):
    insights = []

    # Average spending
    avg_spend = df["Purchase Amount (USD)"].mean()
    insights.append(f"💰 Average customer spending is ${avg_spend:.2f}")

    # Most active cluster
    top_cluster = df["Cluster"].value_counts().idxmax()
    insights.append(
        f"👥 Cluster {top_cluster} contains the highest number of customers, "
        f"indicating the dominant shopper group."
    )

    # High value customers
    high_spenders = df[df["Purchase Amount (USD)"] > df["Purchase Amount (USD)"].mean()]
    insights.append(
        f"🔥 {len(high_spenders)} customers spend above average and are ideal targets for premium offers."
    )

    # Discount behavior
    if "Discount Applied" in df.columns:
        discount_users = df[df["Discount Applied"] == 1]
        insights.append(
            f"🏷️ {len(discount_users)} customers are influenced by discounts, "
            f"suggesting promotions strongly affect purchases."
        )

    # Purchase frequency
    if "Frequency of Purchases" in df.columns:
        freq_avg = df["Frequency of Purchases"].mean()
        insights.append(
            f"🔁 Average purchase frequency is {freq_avg:.1f}, showing moderate repeat buying behavior."
        )

    return insights
