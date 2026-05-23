"""
=============================================================
PatrolIQ - Smart Safety Analytics Platform
=============================================================
Main Streamlit entry point.
Run with:  streamlit run app.py
=============================================================
"""

import streamlit as st

st.set_page_config(
    page_title="PatrolIQ — Smart Safety Analytics",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar nav ───────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/color/96/police-badge.png", width=80)
st.sidebar.title("PatrolIQ")
st.sidebar.caption("Smart Safety Analytics Platform")
st.sidebar.markdown("---")

PAGES = {
    "🏠 Home & Overview"         : "pages/1_Home.py",
    "📊 Exploratory Analysis"     : "pages/2_EDA.py",
    "🗺️ Geographic Clustering"   : "pages/3_Geo_Clustering.py",
    "⏰ Temporal Clustering"      : "pages/4_Temporal_Clustering.py",
    "🔬 Dimensionality Reduction" : "pages/5_Dim_Reduction.py",
    "📈 MLflow Experiment Tracker": "pages/6_Mlflow_Tracker.py",
}

selection = st.sidebar.radio("Navigate to", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.info("**Dataset:** Chicago Crime Data\n\n"
                "**Records:** 494,423 cleaned\n\n"
                "**Features:** 19 engineered")

# ── Route to page ─────────────────────────────────────────
if selection == "🏠 Home & Overview":
    exec(open("pages/1_Home.py", encoding="utf-8").read())
elif selection == "📊 Exploratory Analysis":
    exec(open("pages/2_EDA.py").read())
elif selection == "🗺️ Geographic Clustering":
    exec(open("pages/3_Geo_Clustering.py").read())
elif selection == "⏰ Temporal Clustering":
    exec(open("pages/4_Temporal_Clustering.py").read())
elif selection == "🔬 Dimensionality Reduction":
    exec(open("pages/5_Dim_Reduction.py").read())
elif selection == "📈 MLflow Experiment Tracker":
    exec(open("pages/6_Mlflow_Tracker.py").read())
