import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data_preprocessing import preprocess_data
from src.clustering import perform_clustering
from src.insights_generator import generate_insights

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Deep Shopper Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- SCIFI UI STYLING ----------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;600&display=swap');

    /* Global Background & Typography */
    .stApp {
        background: radial-gradient(circle at center, #0f172a 0%, #000000 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    /* Futuristic Headers */
    h1, h2, h3, .persona-header {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #ffffff;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5), 0 0 20px rgba(0, 212, 255, 0.2);
    }

    /* Cyberpunk Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        border-right: 1px solid #00d4ff;
    }

    /* Glassmorphism Hub Cards */
    div[data-testid="metric-container"], .stPlotlyChart, .persona-card {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        backdrop-filter: blur(15px);
        border-radius: 10px;
        padding: 25px;
        transition: 0.3s ease-in-out;
    }
    
    div[data-testid="metric-container"]:hover {
        border: 1px solid #00d4ff !important;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
    }

    /* Glowing Metrics */
    [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-family: 'Orbitron', sans-serif;
        text-shadow: 0 0 5px #00d4ff;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #000; }
    ::-webkit-scrollbar-thumb { background: #00d4ff; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_and_process():
    try:
        RAW_PATH = "data/raw_data.csv"
        df_scaled, df_original = preprocess_data(RAW_PATH, "data/processed_data.csv")
        clustered_df, _ = perform_clustering(df_scaled)
        df_original["Cluster"] = clustered_df["Cluster"]
        return df_original
    except Exception as e:
        st.error(f"System Linkage Failure: {e}")
        return pd.DataFrame()

df_original = load_and_process()

# ---------------- SIDEBAR INTERFACE ----------------
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>CORE SYSTEM</h2>", unsafe_allow_html=True)
    st.divider()
    
    with st.expander("📡 SENSOR FILTERS", expanded=True):
        gender = st.multiselect("GENDER SOURCE", df_original["Gender"].unique(), default=list(df_original["Gender"].unique()))
        season = st.multiselect("TEMPORAL CYCLE", df_original["Season"].unique(), default=list(df_original["Season"].unique()))
        category = st.multiselect("SECTOR CATEGORY", df_original["Category"].unique(), default=list(df_original["Category"].unique()))

filtered_df = df_original[
    (df_original["Gender"].isin(gender)) &
    (df_original["Category"].isin(category)) &
    (df_original["Season"].isin(season))
]

# ---------------- MAIN INTERFACE ----------------
st.markdown("<h1 style='text-align:center;'>INTRODUCTION TO <br>DEEP INTELLIGENCE</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; opacity:0.6;'>Neural Customer Segmentation & Pattern Recognition</p>", unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("SYSTEM ALERT: No data detected in selected sectors.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["[ 📊 SYSTEM OVERVIEW ]", "[ 🧠 NEURAL INSIGHTS ]", "[ 📂 RAW BYTES ]"])

with tab1:
    m1, m2, m3, m4 = st.columns(4)
    avg_spend = filtered_df["Purchase Amount (USD)"].mean()
    
    m1.metric("NODES (CUSTOMERS)", f"{len(filtered_df):,}")
    m2.metric("AVG THROUGHPUT", f"${avg_spend:.2f}")
    m3.metric("NEURAL CLUSTERS", filtered_df["Cluster"].nunique())
    m4.metric("PEAK CATEGORY", filtered_df["Category"].mode()[0])

    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Cluster Density")
        chart_data = filtered_df["Cluster"].value_counts().reset_index()
        chart_data.columns = ['ID', 'Value']
        fig_bar = px.bar(chart_data, x="ID", y="Value", color="Value",
                         color_continuous_scale="Electric", template="plotly_dark")
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.subheader("Sector Allocation")
        fig_pie = px.pie(filtered_df, names="Cluster", hole=0.6,
                         color_discrete_sequence=px.colors.sequential.Plasma_r, template="plotly_dark")
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.subheader("🧬 Logic Matrix")
    numeric_df = filtered_df.select_dtypes(include=["number"])
    fig_heat = px.imshow(numeric_df.corr(), text_auto=".2f", color_continuous_scale="Viridis", template="plotly_dark")
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🧠 AI SYNAPSE")
        for insight in generate_insights(filtered_df):
            st.success(f"ANALYSIS: {insight}")

    with col_b:
        top_c = filtered_df["Cluster"].value_counts().idxmax()
        st.markdown(f"""
            <div class="persona-card">
                <div class="persona-header">ARCHETYPE: NODE {top_c}</div>
                <p style='color:#00d4ff'><b>INTENSITY:</b> High spending in {category[0] if category else 'various'} sectors.</p>
                <p><b>BEHAVIOR:</b> Responds to high-tech personalization and algorithmic targeting.</p>
            </div>
        """, unsafe_allow_html=True)

with tab3:
    st.dataframe(filtered_df.style.background_gradient(cmap='Blues'), use_container_width=True)
