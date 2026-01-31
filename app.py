import streamlit as st
import os

from src.data_preprocessing import preprocess_data
from src.clustering import perform_clustering
from src.insights_generator import generate_insights

st.set_page_config(page_title="Shopper Behavior Analysis", layout="wide")
st.title("🛍 Shopper Behavior Analysis")

RAW_PATH = "data/raw_data.csv"

df_scaled, df_original = preprocess_data(RAW_PATH, "data/processed_data.csv")

clustered_df, _ = perform_clustering(df_scaled)
df_original["Cluster"] = clustered_df["Cluster"]

st.subheader("Customer Data")
st.dataframe(df_original.head())

st.subheader("AI Insights")
for insight in generate_insights(df_original):
    st.write("•", insight)
