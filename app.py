import streamlit as st
import pandas as pd

from src.data_preprocessing import preprocess_data
from src.clustering import perform_clustering
from src.insights_generator import generate_insights

st.set_page_config(page_title="Shopper Behavior Analysis", layout="wide")

st.title("🛍 Shopper Behavior Analysis Dashboard")

# Load data
raw_data_path = "data/raw_data.csv"
processed_data_path = "data/processed_data.csv"

# Preprocess
df = preprocess_data(raw_data_path, processed_data_path)

# Clustering
clustered_data, _ = perform_clustering(df)

# Display Data
st.subheader("Customer Data Preview")
st.dataframe(clustered_data.head())

# Insights
st.subheader("AI Generated Insights")
insights = generate_insights(clustered_data)
for i in insights:
    st.write("•", i)
