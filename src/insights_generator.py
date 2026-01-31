def generate_insights(df):
    insights = []

    avg_spend = df["Purchase Amount (USD)"].mean()
    insights.append(f"Average customer spending is ${avg_spend:.2f}")

    top_cluster = df["Cluster"].value_counts().idxmax()
    insights.append(f"Cluster {top_cluster} contains the highest number of customers")

    return insights
