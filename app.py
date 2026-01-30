import streamlit as st
import pandas as pd
import os

from src.data_preprocessing import preprocess_data
from src.clustering import perform_clustering
from src.insights_generator import generate_insights

st.set_page_config(page_title="Shopper Behavior Analysis", layout="wide")

st.title("🛍 Shopper Behavior Analysis")

RAW_PATH = "data/raw_data.csv"
PROCESSED_PATH = "data/processed_data.csv"

# Safety check
if not os.path.exists(RAW_PATH):
    st.error("❌ raw_data.csv not found in data folder.")
    st.stop()

# Preprocess
df = preprocess_data(RAW_PATH, PROCESSED_PATH)

# Clustering
clustered_df, _ = perform_clustering(df)

st.subheader("Customer Data Preview")
st.dataframe(clustered_df.head())

st.subheader("AI Generated Insights")
for insight in generate_insights(clustered_df):
    st.write("•", insight)
