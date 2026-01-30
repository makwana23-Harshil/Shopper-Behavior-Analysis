from sklearn.cluster import KMeans

def perform_clustering(df):
    kmeans = KMeans(n_clusters=4, random_state=42)
    df["Cluster"] = kmeans.fit_predict(df)
    return df, kmeans
