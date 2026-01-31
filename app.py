import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from src.data_preprocessing import preprocess_data
from src.clustering import perform_clustering
from src.insights_generator import generate_insights

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Deep Shopper Intelligence",
    page_icon="🧬",
    layout="wide"
)

# ================= CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;600&display=swap');

.stApp {
    background: radial-gradient(circle at center, #0f172a 0%, #000000 100%);
    color: white;
    font-family: 'Inter', sans-serif;
}

h1 {
    font-family: 'Orbitron', sans-serif;
    text-align: center;
    color: white;
    text-shadow: 0 0 15px rgba(0,212,255,0.7);
}

.metric-card {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
}

.persona-card {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df_scaled, df_original = preprocess_data("data/raw_data.csv", "data/processed.csv")
    clustered, _ = perform_clustering(df_scaled)
    df_original["Cluster"] = clustered["Cluster"]
    return df_original

df = load_data()

# ================= SIDEBAR =================
st.sidebar.markdown("## 🔍 Filters")

gender = st.sidebar.multiselect("Gender", df["Gender"].unique(), df["Gender"].unique())
category = st.sidebar.multiselect("Category", df["Category"].unique(), df["Category"].unique())
season = st.sidebar.multiselect("Season", df["Season"].unique(), df["Season"].unique())

filtered = df[
    (df["Gender"].isin(gender)) &
    (df["Category"].isin(category)) &
    (df["Season"].isin(season))
]

# SAFETY CHECK
if filtered.empty:
    st.warning("⚠️ No data available for selected filters.")
    st.stop()

# ================= HEADER =================
st.markdown("<h1>🧬 Deep Shopper Intelligence</h1>", unsafe_allow_html=True)

tabs = st.tabs(["📊 Overview", "🧠 Intelligence", "📁 Data"])

# ================= OVERVIEW =================
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Customers", len(filtered))
    c2.metric("Avg Spend", f"${filtered['Purchase Amount (USD)'].mean():.2f}")
    c3.metric("Clusters", filtered["Cluster"].nunique())
    c4.metric("Top Category", filtered["Category"].mode()[0])

    colA, colB = st.columns(2)

    with colA:
        fig1 = px.bar(
            filtered["Cluster"].value_counts().reset_index(),
            x="index", y="Cluster",
            color="Cluster",
            template="plotly_dark",
            title="Customer Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with colB:
        fig2 = px.pie(
            filtered,
            names="Cluster",
            hole=0.5,
            template="plotly_dark",
            title="Cluster Share"
        )
        st.plotly_chart(fig2, use_container_width=True)

# ================= INTELLIGENCE =================
with tabs[1]:
    st.subheader("🧠 Correlation Matrix")

    corr = filtered.select_dtypes("number").corr()
    fig = px.imshow(
        corr,
        color_continuous_scale="viridis",
        text_auto=True,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    colL, colR = st.columns(2)

    with colL:
        st.subheader("AI Insights")
        for i in generate_insights(filtered):
            st.success(i)

    with colR:
        top_cluster = filtered["Cluster"].value_counts().idxmax()
        st.markdown(f"""
        <div class="persona-card">
            <h3>👤 Customer Persona</h3>
            <p><b>Cluster:</b> {top_cluster}</p>
            <p><b>Behavior:</b> High engagement, value-driven</p>
            <p><b>Strategy:</b> Target with loyalty & personalization</p>
        </div>
        """, unsafe_allow_html=True)

# ================= RAW DATA =================
with tabs[2]:
    st.dataframe(filtered, use_container_width=True)

    st.download_button(
        "⬇ Download CSV",
        filtered.to_csv(index=False),
        "filtered_data.csv",
        "text/csv"
    )

# ================= FOOTER =================
st.markdown(
    f"<div style='text-align:center;opacity:0.5;margin-top:30px;'>"
    f"{datetime.now().strftime('%d %B %Y')}</div>",
    unsafe_allow_html=True
)
