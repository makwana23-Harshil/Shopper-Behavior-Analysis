import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_preprocessing import preprocess_data
from src.clustering import perform_clustering
from src.insights_generator import generate_insights

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Shopper Behavior Analytics",
    layout="wide",
    page_icon="🛍️"
)

# ---------------- DARK MODE TOGGLE ----------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown(
        """
        <style>
        body { background-color: #0f172a; color: white; }
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center;'>🛍 Shopper Behavior Analysis</h1>
    <p style='text-align: center; font-size:18px;'>
    AI-powered customer segmentation & insights dashboard
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD DATA ----------------
RAW_PATH = "data/raw_data.csv"
df_scaled, df_original = preprocess_data(RAW_PATH, "data/processed_data.csv")
clustered_df, _ = perform_clustering(df_scaled)
df_original["Cluster"] = clustered_df["Cluster"]

# ---------------- KPI METRICS ----------------
st.markdown("## 📌 Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("👥 Total Customers", len(df_original))

with col2:
    avg_spend = df_original["Purchase Amount (USD)"].mean()
    st.metric("💰 Avg Spending", f"${avg_spend:.2f}")

with col3:
    st.metric("🧩 Clusters", df_original["Cluster"].nunique())

# ---------------- CHARTS ----------------
st.markdown("## 📊 Visual Insights")

col4, col5 = st.columns(2)

# 🔹 Bar Chart
with col4:
    st.subheader("Customer Segments")
    fig1, ax1 = plt.subplots()
    df_original["Cluster"].value_counts().plot(
        kind="bar", ax=ax1, color="#3b82f6"
    )
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("Customers")
    st.pyplot(fig1)

# 🔹 Pie Chart
with col5:
    st.subheader("Cluster Distribution")
    fig2, ax2 = plt.subplots()
    df_original["Cluster"].value_counts().plot(
        kind="pie", autopct="%1.1f%%", ax=ax2
    )
    ax2.set_ylabel("")
    st.pyplot(fig2)

# ---------------- HEATMAP ----------------
st.markdown("## 🔥 Correlation Heatmap")

numeric_df = df_original.select_dtypes(include=["int64", "float64"])

fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)

# ---------------- AI INSIGHTS ----------------
st.markdown("## 🧠 AI Insights")

for insight in generate_insights(df_original):
    st.markdown(f"✅ {insight}")

# ---------------- DATA PREVIEW ----------------
with st.expander("📂 View Sample Data"):
    st.dataframe(df_original.head(10))
