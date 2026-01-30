import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("data/processed_data.csv")

kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(df)

df.to_csv("data/clustered_data.csv", index=False)
print("✅ Clustering completed")

