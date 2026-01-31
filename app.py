import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_preprocessing import preprocess_data
from src.clustering import perform_clustering
from src.insights_generator import generate_insights

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Shopper Behavior Analytics",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
df_scaled, df_original = preprocess_data(
    "data/raw_data.csv",
    "data/processed.csv"
)

clustered_df, _ = perform_clustering(df_scaled)
df_original["Cluster"] = clustered_df["Cluster"]

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.title("Filters")

gender = st.sidebar.multiselect(
    "Gender",
    options=df_original["Gender"].unique(),
    default=df_original["Gender"].unique()
)

category = st.sidebar.multiselect(
    "Category",
    options=df_original["Category"].unique(),
    default=df_original["Category"].unique()
)

season = st.sidebar.multiselect(
    "Season",
    options=df_original["Season"].unique(),
    default=df_original["Season"].unique()
)

filtered_df = df_original[
    (df_original["Gender"].isin(gender)) &
    (df_original["Category"].isin(category)) &
    (df_original["Season"].isin(season))
]

# ---------------- HEADER ----------------
st.title("🛍 Shopper Behavior Analysis")
st.caption("Customer segmentation and behavioral insights dashboard")

if filtered_df.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# ---------------- KPI METRICS ----------------
st.subheader("Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(filtered_df))
col2.metric("Average Spending ($)", round(filtered_df["Purchase Amount (USD)"].mean(), 2))
col3.metric("Customer Segments", filtered_df["Cluster"].nunique())

# ---------------- CHARTS ----------------
st.subheader("Customer Distribution")

col4, col5 = st.columns(2)

with col4:
    fig1 = px.bar(
        filtered_df,
        x="Cluster",
        title="Customers per Cluster",
        color="Cluster",
        template="plotly_white"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col5:
    fig2 = px.pie(
        filtered_df,
        names="Cluster",
        title="Cluster Share",
        hole=0.4,
        template="plotly_white"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- HEATMAP ----------------
st.subheader("Correlation Analysis")

numeric_df = filtered_df.select_dtypes(include=["int64", "float64"])
fig3 = px.imshow(
    numeric_df.corr(),
    color_continuous_scale="Blues",
    title="Feature Correlation Heatmap"
)
st.plotly_chart(fig3, use_container_width=True)

# ---------------- AI INSIGHTS ----------------
st.subheader("AI Insights")

for insight in generate_insights(filtered_df):
    st.success(insight)

# ---------------- DATA TABLE ----------------
st.subheader("Filtered Data Preview")

st.dataframe(filtered_df, use_container_width=True)

st.download_button(
    "Download CSV",
    filtered_df.to_csv(index=False),
    "filtered_data.csv",
    "text/csv"
)
