import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
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

# ---------------- HUD UI STYLING ----------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=Inter:wght@300;600&display=swap');

    /* Background: Deep Cosmic Radial */
    .stApp {
        background: radial-gradient(circle at center, #0f172a 0%, #000000 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    /* Glowing Sci-Fi Headers */
    h1 {
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 900 !important;
        letter-spacing: -1px !important;
        text-transform: uppercase;
        color: #ffffff !important;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.8), 0 0 40px rgba(0, 212, 255, 0.4) !important;
        font-size: 3.5rem !important;
        text-align: center;
        margin-bottom: 0px !important;
    }
    
    .sub-text {
        text-align: center;
        font-size: 0.8rem;
        letter-spacing: 0.5rem;
        text-transform: uppercase;
        color: rgba(255, 255, 255, 0.5);
        margin-bottom: 40px;
    }

    /* Glassmorphism Hub Cards */
    div[data-testid="metric-container"], .stPlotlyChart, .persona-card {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    }

    /* Sidebar HUD Look */
    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0, 212, 255, 0.2);
    }

    /* Tab Styling: Rounded Pill HUD */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 50px !important;
        padding: 8px 30px !important;
        background: rgba(255,255,255,0.02) !important;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.7rem;
    }
    .stTabs [aria-selected="true"] {
        border-color: #00d4ff !important;
        color: #00d4ff !important;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
    }

    /* Footer Pill */
    .footer-pill {
        display: flex;
        justify-content: center;
        margin-top: 50px;
    }
    .pill-content {
        padding: 8px 25px;
        border-radius: 50px;
        border: 1px solid rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.05);
        font-size: 0.7rem;
        color: rgba(255,255,255,0.6);
        letter-spacing: 2px;
    }
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

# ---------------- SIDEBAR NAVIGATION ----------------
with st.sidebar:
    st.markdown("<h3 style='text-align:center; font-family:Orbitron;'>SYSTEM CORE</h3>", unsafe_allow_html=True)
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

# ---------------- MAIN VIEWPORT ----------------
st.markdown("<p class='sub-text'>Introduction To</p>", unsafe_allow_html=True)
st.markdown("<h1>Deep Learning</h1>", unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("SYSTEM ALERT: No data signature detected in current sectors.")
    st.stop()

tab_overview, tab_insights, tab_raw = st.tabs(["[ 📊 DATA OVERVIEW ]", "[ 🧠 NEURAL ANALYSIS ]", "[ 💾 DATA EXPORT ]"])

with tab_overview:
    # KPI Grid
    m1, m2, m3, m4 = st.columns(4)
    avg_spend = filtered_df["Purchase Amount (USD)"].mean()
    
    m1.metric("TOTAL NODES", f"{len(filtered_df):,}")
    m2.metric("AVG THROUGHPUT", f"${avg_spend:.2f}")
    m3.metric("ACTIVE CLUSTERS", filtered_df["Cluster"].nunique())
    m4.metric("PEAK SECTOR", filtered_df["Category"].mode()[0])

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualization HUD
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Cluster Density Matrix")
        chart_data = filtered_df["Cluster"].value_counts().reset_index()
        chart_data.columns = ['Cluster_ID', 'Value']
        fig_bar = px.bar(chart_data, x="Cluster_ID", y="Value", color="Value",
                         color_continuous_scale="Electric", template="plotly_dark")
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.subheader("Neural Distribution")
        fig_pie = px.pie(filtered_df, names="Cluster", hole=0.6,
                         color_discrete_sequence=px.colors.sequential.Plasma_r, template="plotly_dark")
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)

with tab_insights:
    st.subheader("🧬 Correlation Synapse")
    numeric_df = filtered_df.select_dtypes(include=["number"])
    fig_heat = px.imshow(numeric_df.corr(), text_auto=".2f", color_continuous_scale="Viridis", template="plotly_dark")
    st.plotly_chart(fig_heat, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🧠 Logic Output")
        for insight in generate_insights(filtered_df):
            st.info(f"SYNAPSE: {insight}")

    with col_b:
        top_c = filtered_df["Cluster"].value_counts().idxmax()
        st.markdown(f"""
            <div class="persona-card">
                <h3 style='color:#00d4ff; margin-top:0;'>ARCHETYPE: NODE {top_c}</h3>
                <p><b>Spending Priority:</b> High concentration in {category[0] if category else 'diverse'} sectors.</p>
                <p><b>Strategic Vector:</b> Target with algorithmic precision and personalized rewards.</p>
            </div>
        """, unsafe_allow_html=True)

with tab_raw:
    st.markdown("### 📂 Filtered Byte Data")
    st.dataframe(filtered_df.style.background_gradient(cmap='Blues'), use_container_width=True)
    
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(label="⬇️ EXPORT DATASTREAM", data=csv, file_name="shopper_export.csv", mime="text/csv")

# ---------------- FOOTER ----------------
st.markdown(f"""
    <div class="footer-pill">
        <div class="pill-content">
            {datetime.now().strftime('%d %B, %Y')}
        </div>
    </div>
""", unsafe_allow_html=True)
