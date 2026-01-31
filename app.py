import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data_preprocessing import preprocess_data
from src.clustering import perform_clustering
from src.insights_generator import generate_insights

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Shopper Behavior Analytics",
    layout="wide",
    page_icon="🛍️"
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center;'>🛍 Shopper Behavior Analysis</h1>
    <p style='text-align: center; font-size:18px;'>
    AI-powered customer segmentation & purchasing insights
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD DATA ----------------
RAW_PATH = "data/raw_data.csv"
df_scaled, df_original = preprocess_data(RAW_PATH, "data/processed_data.csv")
clustered_df, _ = perform_clustering(df_scaled)
df_original["Cluster"] = clustered_df["Cluster"]

# ---------------- KPI SECTION ----------------
st.markdown("## 📌 Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", len(df_original))

with col2:
    avg_spend = df_original["Purchase Amount (USD)"].mean()
    st.metric("Avg Spending ($)", f"{avg_spend:.2f}")

with col3:
    st.metric("Total Clusters", df_original["Cluster"].nunique())

# ---------------- CHARTS ----------------
st.markdown("## 📊 Visual Insights")

col4, col5 = st.columns(2)

# Cluster Distribution
with col4:
    st.markdown("### Customer Segments")
    fig1, ax1 = plt.subplots()
    df_original["Cluster"].value_counts().plot(
        kind="bar",
        ax=ax1,
        color="#4F8BF9"
    )
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("Customers")
    st.pyplot(fig1)

# Spending by Cluster
with col5:
    st.markdown("### Avg Spending per Cluster")
    avg_cluster_spend = df_original.groupby("Cluster")["Purchase Amount (USD)"].mean()
    fig2, ax2 = plt.subplots()
    avg_cluster_spend.plot(kind="bar", ax=ax2, color="#22C55E")
    ax2.set_ylabel("USD")
    st.pyplot(fig2)

# ---------------- AI INSIGHTS ----------------
st.markdown("## 🧠 AI Insights")

for insight in generate_insights(df_original):
    st.markdown(f"✅ {insight}")

# ---------------- DATA PREVIEW ----------------
with st.expander("📂 View Sample Data"):
    st.dataframe(df_original.head(10))
