from sklearn.cluster import KMeans

def perform_clustering(df):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    # Safety check
    if numeric_df.empty:
        raise ValueError("No numeric columns available for clustering.")

    # Apply KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(numeric_df)

    # Add cluster labels back to original dataframe
    df["Cluster"] = clusters

    return df, kmeans
