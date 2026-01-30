import streamlit as st
import pandas as pd

st.title("🛍 Shopper Behavior Analysis")

data = pd.read_csv("data/clustered_data.csv")

st.subheader("Customer Data")
st.dataframe(data.head())

st.subheader("Cluster Distribution")
st.bar_chart(data["Cluster"].value_counts())

