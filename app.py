import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data_preprocessing import preprocess_data
from src.clustering import perform_clustering
from src.insights_generator import generate_insights

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Shopper Behavior Analytics", layout="wide")

# ---------------- DARK MODE ----------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown("""
        <style>
        .stApp { background-color: #0f172a; color: white; }
        </style>
    """, unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>🛍 Shopper Behavior Analysis</h1>", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
RAW_PATH = "data/raw_data.csv"
df_scaled, df_original = preprocess_data(RAW_PATH, "data/processed_data.csv")
clustered_df, _ = perform_clustering(df_scaled)
df_original["Cluster"] = clustered_df["Cluster"]

# ---------------- FILTERS ----------------
st.sidebar.header("🔍 Filters")

gender = st.sidebar.multiselect(
    "Gender", df_original["Gender"].unique(), default=df_original["Gender"].unique()
)

category = st.sidebar.multiselect(
    "Category", df_original["Category"].unique(), default=df_original["Category"].unique()
)

season = st.sidebar.multiselect(
    "Season", df_original["Season"].unique(), default=df_original["Season"].unique()
)

filtered_df = df_original[
    (df_original["Gender"].isin(gender)) &
    (df_original["Category"].isin(category)) &
    (df_original["Season"].isin(season))
]

# ---------------- HANDLE EMPTY DATA ----------------
if filtered_df.empty:
    st.warning("⚠️ No data available for selected filters. Please change filters.")
    st.stop()

# ---------------- KPI SECTION ----------------
st.markdown("## 📌 Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("👥 Customers", len(filtered_df))

with col2:
    avg_spend = filtered_df["Purchase Amount (USD)"].mean()
    st.metric("💰 Avg Spending", f"${avg_spend:.2f}")

with col3:
    st.metric("🧩 Clusters", filtered_df["Cluster"].nunique())

# ---------------- CHARTS ----------------
st.markdown("## 📊 Visual Insights")

col4, col5 = st.columns(2)

# Bar Chart
with col4:
    st.subheader("Customer Segments")
    fig1, ax1 = plt.subplots()
    filtered_df["Cluster"].value_counts().plot(
        kind="bar", ax=ax1, color="#3b82f6"
    )
    ax1.set_ylabel("Customers")
    st.pyplot(fig1)

# Pie Chart
with col5:
    st.subheader("Cluster Distribution")
    fig2, ax2 = plt.subplots()
    filtered_df["Cluster"].value_counts().plot(
        kind="pie", autopct="%1.1f%%", ax=ax2
    )
    ax2.set_ylabel("")
    st.pyplot(fig2)

# ---------------- HEATMAP ----------------
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

st.markdown("## 📥 Download Filtered Data")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download CSV",
    data=csv,
    file_name="filtered_customer_data.csv",
    mime="text/csv"
)

# ---------------- AI INSIGHTS ----------------
st.markdown("## 🧠 AI Insights")

for insight in generate_insights(filtered_df):
    st.markdown(f"✅ {insight}")

# ---------------- DATA PREVIEW ----------------
with st.expander("📂 View Filtered Data"):
    st.dataframe(filtered_df.head(10))

#-------------------Auto summary -----------------------------------
st.markdown("## 🧠 Auto Summary")

avg_spend = filtered_df["Purchase Amount (USD)"].mean()
top_cluster = filtered_df["Cluster"].value_counts().idxmax()
total_customers = len(filtered_df)

summary_text = f"""
This dataset contains **{total_customers} customers**.  
The **average spending is ${avg_spend:.2f}**, indicating moderate purchasing behavior.  
**Cluster {top_cluster}** represents the dominant customer group, suggesting a major opportunity for targeted marketing.  
Overall, customer behavior shows clear segmentation patterns that can be used for personalization, promotions, and retention strategies.
"""

st.info(summary_text)

#__________________________
st.markdown("## 👤 Customer Persona")

persona = f"""
### 🧍 Typical Customer Profile

• **Spending Behavior:** Moderate spender  
• **Shopping Pattern:** Belongs to Cluster {top_cluster}  
• **Price Sensitivity:** Responds well to discounts  
• **Engagement Level:** Medium to High  
• **Business Insight:** Best target for loyalty programs and personalized offers
"""

st.success(persona)
