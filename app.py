import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_preprocessing import preprocess_data
from src.clustering import perform_clustering
from src.insights_generator import generate_insights

# ===================== CONFIG =====================
st.set_page_config(
    page_title="Shopper Intelligence",
    page_icon="🛍️",
    layout="wide"
)

# ===================== CSS (GLASSMORPHISM) =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

.glass {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}

.metric-card {
    background: linear-gradient(135deg, #6366f1, #ec4899);
    padding: 20px;
    border-radius: 16px;
    color: white;
    text-align: center;
}

.sidebar .sidebar-content {
    background: linear-gradient(180deg, #020617, #0f172a);
}
</style>
""", unsafe_allow_html=True)

# ===================== DATA =====================
RAW_PATH = "data/raw_data.csv"
df_scaled, df_original = preprocess_data(RAW_PATH, "data/processed.csv")
df_clustered, _ = perform_clustering(df_scaled)
df_original["Cluster"] = df_clustered["Cluster"]

# ===================== SIDEBAR =====================
st.sidebar.markdown("## 🔍 Filters")

gender = st.sidebar.multiselect("Gender", df_original["Gender"].unique(),
                                 default=df_original["Gender"].unique())

category = st.sidebar.multiselect("Category", df_original["Category"].unique(),
                                   default=df_original["Category"].unique())

season = st.sidebar.multiselect("Season", df_original["Season"].unique(),
                                 default=df_original["Season"].unique())

filtered = df_original[
    (df_original["Gender"].isin(gender)) &
    (df_original["Category"].isin(category)) &
    (df_original["Season"].isin(season))
]

# ===================== HEADER =====================
st.markdown("""
<h1 style='text-align:center;'>🛍 Shopper Behavior Intelligence</h1>
<p style='text-align:center;'>AI-powered consumer analytics dashboard</p>
""", unsafe_allow_html=True)

# ===================== TABS =====================
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Deep Analytics", "📁 Raw Data"])

# ======================================================
# ===================== TAB 1 ==========================
# ======================================================
with tab1:
    st.markdown("### 📌 Key Metrics")

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"""
    <div class='metric-card'>
        <h3>{len(filtered)}</h3>
        <p>Total Customers</p>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class='metric-card'>
        <h3>${filtered['Purchase Amount (USD)'].mean():.2f}</h3>
        <p>Avg Spending</p>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class='metric-card'>
        <h3>{filtered['Cluster'].nunique()}</h3>
        <p>Customer Segments</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- CHARTS ----------------
    st.markdown("### 📊 Customer Distribution")

    fig1 = px.bar(
        filtered,
        x="Cluster",
        title="Customer Segments",
        color="Cluster",
        template="plotly_dark"
    )

    fig2 = px.pie(
        filtered,
        names="Cluster",
        title="Cluster Share",
        hole=0.45,
        template="plotly_dark"
    )

    col4, col5 = st.columns(2)
    col4.plotly_chart(fig1, use_container_width=True)
    col5.plotly_chart(fig2, use_container_width=True)

# ======================================================
# ===================== TAB 2 ==========================
# ======================================================
with tab2:
    st.markdown("### 🔥 Correlation Heatmap")

    corr = filtered.select_dtypes("number").corr()

    heatmap = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="viridis",
        title="Feature Correlation Matrix"
    )

    st.plotly_chart(heatmap, use_container_width=True)

    # ---------------- PERSONA CARD ----------------
    st.markdown("### 👤 Customer Persona")

    top_cluster = filtered["Cluster"].value_counts().idxmax()

    st.markdown(f"""
    <div class='glass'>
        <h3>🎯 Primary Customer Persona</h3>
        <p><b>Cluster:</b> {top_cluster}</p>
        <p><b>Spending:</b> Moderate to High</p>
        <p><b>Behavior:</b> Repeat buyer, promotion responsive</p>
        <p><b>Business Insight:</b> Ideal for loyalty & upselling campaigns</p>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# ===================== TAB 3 ==========================
# ======================================================
with tab3:
    st.markdown("### 📁 Raw Dataset")
    st.dataframe(filtered, use_container_width=True)

    st.download_button(
        "⬇ Download CSV",
        data=filtered.to_csv(index=False),
        file_name="filtered_data.csv",
        mime="text/csv"
    )

# ===================== AI INSIGHTS =====================
st.markdown("## 🧠 AI Insights")

for insight in generate_insights(filtered):
    st.success(insight)
