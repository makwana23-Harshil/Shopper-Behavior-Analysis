import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data_preprocessing import preprocess_data
from src.clustering import perform_clustering
from src.insights_generator import generate_insights

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Shopper Intelligence | Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM UI STYLING ----------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #1e293b, #0f172a);
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background-image: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    div[data-testid="metric-container"], .stPlotlyChart, .persona-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }

    .persona-card {
        border-left: 5px solid #6366f1;
    }
    
    .persona-header {
        color: #6366f1;
        font-weight: 800;
        margin-bottom: 10px;
    }

    [data-testid="stMetricValue"] {
        font-weight: 700;
        color: #6366f1;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_and_process():
    RAW_PATH = "data/raw_data.csv"
    df_scaled, df_original = preprocess_data(RAW_PATH, "data/processed_data.csv")
    clustered_df, _ = perform_clustering(df_scaled)
    df_original["Cluster"] = clustered_df["Cluster"]
    return df_original

df_original = load_and_process()

# ---------------- SIDEBAR FILTERS ----------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=80)
    st.title("Filters")
    
    with st.expander("🎯 Target Demographics", expanded=True):
        gender = st.multiselect("Gender", df_original["Gender"].unique(), default=list(df_original["Gender"].unique()))
        season = st.multiselect("Season", df_original["Season"].unique(), default=list(df_original["Season"].unique()))

    with st.expander("📦 Product Categories", expanded=True):
        category = st.multiselect("Category", df_original["Category"].unique(), default=list(df_original["Category"].unique()))

filtered_df = df_original[
    (df_original["Gender"].isin(gender)) &
    (df_original["Category"].isin(category)) &
    (df_original["Season"].isin(season))
]

# ---------------- HEADER ----------------
st.markdown("# 🛍️ Shopper Behavior Intelligence")
st.markdown("Analyze customer segmentation and purchasing patterns with AI-driven insights.")

if filtered_df.empty:
    st.warning("⚠️ No data matches your filter criteria.")
    st.stop()

# ---------------- MAIN TABS ----------------
tab_overview, tab_deep_dive, tab_data = st.tabs(["📈 Overview", "🧠 AI & Personas", "💾 Raw Data"])

with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    avg_spend = filtered_df["Purchase Amount (USD)"].mean()
    
    col1.metric("Total Customers", f"{len(filtered_df):,}")
    col2.metric("Avg Spending", f"${avg_spend:.2f}")
    col3.metric("Active Clusters", filtered_df["Cluster"].nunique())
    col4.metric("Top Category", filtered_df["Category"].mode()[0])

    st.markdown("---")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Customer Segments")
        # FIXED: Explicitly naming columns to avoid ValueError
        chart_data = filtered_df["Cluster"].value_counts().reset_index()
        chart_data.columns = ['Cluster_ID', 'Count'] 

        fig_bar = px.bar(
            chart_data,
            x="Cluster_ID", 
            y="Count",
            labels={"Cluster_ID": "Cluster Group", "Count": "Customer Count"},
            color="Cluster_ID",
            color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.subheader("Distribution %")
        fig_pie = px.pie(
            filtered_df, names="Cluster",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.RdBu,
            template="plotly_dark"
        )
        fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("🔥 Feature Correlation Matrix")
    numeric_df = filtered_df.select_dtypes(include=["number"])
    corr = numeric_df.corr()
    fig_heat = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        template="plotly_dark"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with tab_deep_dive:
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.markdown("### 🧠 AI Strategic Insights")
        for insight in generate_insights(filtered_df):
            st.info(f"💡 {insight}")

    with col_b:
        st.markdown("### 👤 Customer Persona")
        top_cluster = filtered_df["Cluster"].value_counts().idxmax()
        st.markdown(f"""
            <div class="persona-card">
                <div class="persona-header">PRIMARY ARCHETYPE: Cluster {top_cluster}</div>
                <p><b>Spending Profile:</b> ${avg_spend:.2f} (Average)</p>
                <p><b>Psychographics:</b> High engagement in {season[0] if season else 'all seasons'}.</p>
                <p><b>Recommendation:</b> Deploy loyalty rewards and personalized email triggers 
                based on {category[0] if category else 'general'} purchase history.</p>
            </div>
        """, unsafe_allow_html=True)

with tab_data:
    st.markdown("### 📂 Filtered Dataset")
    st.dataframe(filtered_df, use_container_width=True)
    
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(label="⬇️ Export Data", data=csv, file_name="export.csv", mime="text/csv")
