import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data_preprocessing import preprocess_data
from src.clustering import perform_clustering
from src.insights_generator import generate_insights

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(
    page_title="Shopper Behavior Analytics",
    layout="wide",
    page_icon="🛍️"
)

# =======================
# DARK MODE TOGGLE (REAL)
# =======================
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown("""
        <style>
        .stApp { background-color: #0f172a; color: white; }
        h1, h2, h3, h4, h5, h6 { color: #f8fafc; }
        .stMetric { background-color: #1e293b; padding: 10px; border-radius: 10px; }
        </style>
    """, unsafe_allow_html=True)

# =======================
# HEADER
# =======================
st.markdown(
    "<h1 style='text-align:center;'>🛍 Shopper Behavior Analysis</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>AI-powered customer segmentation dashboard</p>",
    unsafe_allow_html=True
)

# =======================
# LOAD DATA
# =======================
RAW_PATH = "data/raw_data.csv"
df_scaled, df_original = preprocess_data(RAW_PATH, "data/processed_data.csv")
clustered_df, _ = perform_clustering(df_scaled)
df_original["Cluster"] = clustered_df["Cluster"]

# =======================
# FILTERS
# =======================
st.sidebar.header("🔍 Filters")

gender_filter = st.sidebar.multiselect(
    "Select Gender",
    options=df_original["Gender"].unique(),
    default=df_original["Gender"].unique()
)

category_filter = st.sidebar.multiselect(
    "Select Category",
    options=df_original["Category"].unique(),
    default=df_original["Category"].unique()
)

season_filter = st.sidebar.multiselect(
    "Select Season",
    options=df_original["Season"].unique(),
    default=df_original["Season"].unique()
)

# Apply filters
filtered_df = df_original[
    (df_original["Gender"].isin(gender_filter)) &
    (df_original["Category"].isin(category_filter)) &
    (df_original["Season"].isin(season_filter))
]

# =======================
# KPI SECTION
# =======================
st.markdown("## 📌 Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("👥 Customers", len(filtered_df))

with col2:
    st.metric("💰 Avg Spending", f"${filtered_df['Purchase Amount (USD)'].mean():.2f}")

with col3:
    st.metric("🧩 Clusters", filtered_df["Cluster"].nunique())

# =======================
# CHARTS
# =======================
st.markdown("## 📊 Visual Insights")

col4, col5 = st.columns(2)

# Bar Chart
with col4:
    st.subheader("Customer Segments")
    fig1, ax1 = plt.subplots()
    filtered_df["Cluster"].value_counts().plot(
        kind="bar",
        ax=ax1,
        color="#3b82f6"
    )
    ax1.set_ylabel("Customers")
    st.pyplot(fig1)

# Pie Chart
with col5:
    st.subheader("Cluster Distribution")
    fig2, ax2 = plt.subplots()
    filtered_df["Cluster"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax2
    )
    ax2.set_ylabel("")
    st.pyplot(fig2)

# =======================
# HEATMAP (No seaborn)
# =======================
st.markdown("## 🔥 Correlation Heatmap")

numeric_df = filtered_df.select_dtypes(include=["int64", "float64"])
corr = numeric_df.corr()

fig3, ax3 = plt.subplots(figsize=(10, 6))
im = ax3.imshow(corr, cmap="coolwarm")
ax3.set_xticks(range(len(corr.columns)))
ax3.set_yticks(range(len(corr.columns)))
ax3.set_xticklabels(corr.columns, rotation=45, ha="right")
ax3.set_yticklabels(corr.columns)
plt.colorbar(im)
st.pyplot(fig3)

# =======================
# AI INSIGHTS
# =======================
st.markdown("## 🧠 AI Insights")

for insight in generate_insights(filtered_df):
    st.markdown(f"✅ {insight}")

# =======================
# DATA PREVIEW
# =======================
with st.expander("📂 View Filtered Data"):
    st.dataframe(filtered_df.head(10))
