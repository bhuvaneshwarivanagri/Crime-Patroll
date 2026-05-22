"""Page 1 — Home & Project Overview"""
import streamlit as st
import pandas as pd

st.title("🚔 PatrolIQ — Smart Safety Analytics Platform")
st.markdown("""
> *Making Chicago safer through data-driven policing intelligence.*

This platform analyzes **494,423 crime records** from the Chicago Police Department
using unsupervised machine learning to:
- Identify geographic crime hotspots
- Discover temporal crime patterns
- Reduce complex crime data into actionable 2D/3D visualizations
- Optimize police patrol resource allocation
""")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records",    "494,423")
col2.metric("Crime Types",      "32")
col3.metric("Districts",        "24")
col4.metric("Date Range",       "2001–2026")

st.markdown("---")
st.subheader("📁 Project Pipeline")

pipeline = [
    ("Step 1", "Data Preprocessing",        "✅ Complete", "Cleaned, extracted temporal features"),
    ("Step 2", "Exploratory Data Analysis",  "✅ Complete", "Charts: crime types, time, geography"),
    ("Step 3", "Feature Engineering",        "✅ Complete", "19 features: encoded, scaled, binned"),
    ("Step 4", "Clustering Analysis",        "✅ Complete", "K-Means, DBSCAN, Hierarchical + Temporal"),
    ("Step 5", "Dimensionality Reduction",   "✅ Complete", "PCA, t-SNE, UMAP"),
    ("Step 6", "Streamlit Application",      "✅ Running",  "This app!"),
    ("Step 7", "MLflow Tracking",            "✅ Complete", "All experiments logged"),
]

df_pipe = pd.DataFrame(pipeline,
    columns=["Step", "Stage", "Status", "Details"])
st.dataframe(df_pipe, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("🔑 Key Findings")
c1, c2 = st.columns(2)
with c1:
    st.info("**Top Crime:** THEFT (104,798 incidents)")
    st.info("**Peak Hour:** Noon (12:00) and Midnight (00:00)")
    st.info("**Highest Risk Season:** Summer")
with c2:
    st.warning("**Arrest Rate:** Only 25.2% of crimes lead to arrest")
    st.warning("**Domestic Incidents:** 17.4% of all crimes")
    st.success("**Best Clustering Algorithm:** K-Means (Silhouette=0.487)")

st.markdown("---")
st.caption("PatrolIQ | GUVI × HCL Capstone Project | Built with Python, Streamlit, scikit-learn, MLflow")
