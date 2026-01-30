import streamlit as st
import pandas as pd
import os

from src.data_preprocessing import preprocess_data
from src.clustering import perform_clustering
from src.insights_generator import generate_insights

st.set_page_config(page_title="Shopper Behavior Analysis", layout="wide")

st.title("🛍 Shopper Behavior Analysis")

DATA_PATH = "data/raw_data.csv"

# Check if file exists
if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset not found! Please upload raw_data.csv in data folder.")
    st.stop()

# Run preprocessing
df = preprocess_data(DATA_PATH, "data/processed_data.csv")

# Clustering
clustered_data, _ = perform_clustering(df)

st.subheader("Customer Data Preview")
st.dataframe(clustered_data.head())

st.subheader("AI Generated Insights")
for insight in generate_insights(clustered_data):
    st.write("•", insight)
