from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

def perform_clustering(df):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    # 🧠 Handle missing values safely
    imputer = SimpleImputer(strategy="median")
    numeric_df_imputed = imputer.fit_transform(numeric_df)

    # Apply KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(numeric_df_imputed)

    # Add cluster labels to original dataframe
    df["Cluster"] = clusters

    return df, kmeans
