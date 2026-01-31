import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt

from src.data_preprocessing import preprocess_data
from src.clustering import perform_clustering
from src.insights_generator import generate_insights

st.set_page_config(page_title="Shopper Behavior Analysis", layout="wide")

st.title("🛍 Shopper Behavior Analysis Dashboard")

RAW_PATH = "data/raw_data.csv"

# Load and preprocess
df_scaled, df_original = preprocess_data(RAW_PATH, "data/processed_data.csv")

# Clustering
clustered_df, _ = perform_clustering(df_scaled)
df_original["Cluster"] = clustered_df["Cluster"]

# ======================
# 📊 VISUALIZATION AREA
# ======================

st.subheader("📊 Cluster Distribution")
cluster_counts = df_original["Cluster"].value_counts()

fig1, ax1 = plt.subplots()
ax1.bar(cluster_counts.index.astype(str), cluster_counts.values)
ax1.set_xlabel("Cluster")
ax1.set_ylabel("Number of Customers")
st.pyplot(fig1)

# ----------------------

st.subheader("💰 Average Spending per Cluster")
avg_spend = df_original.groupby("Cluster")["Purchase Amount (USD)"].mean()

fig2, ax2 = plt.subplots()
ax2.bar(avg_spend.index.astype(str), avg_spend.values)
ax2.set_ylabel("Average Spend ($)")
st.pyplot(fig2)

# ----------------------

if "Frequency of Purchases" in df_original.columns:
    st.subheader("🔁 Purchase Frequency Distribution")
    fig3, ax3 = plt.subplots()
    df_original["Frequency of Purchases"].value_counts().plot(kind="bar", ax=ax3)
    st.pyplot(fig3)

# ======================
# 🧠 AI INSIGHTS
# ======================

st.subheader("🧠 AI Insights")
for insight in generate_insights(df_original):
    st.write("•", insight)
