"""Page 3 — Geographic Clustering"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

st.title("🗺️ Geographic Crime Hotspot Clustering")

@st.cache_data
def load_data():
    X    = pd.read_csv("data/features.csv")
    meta = pd.read_csv("data/metadata.csv", parse_dates=["Date"])
    geo  = pd.read_csv("data/geo_cluster_results.csv")
    return X, meta, geo

X, meta, geo_result = load_data()

st.markdown("""
Geographic clustering identifies **crime hotspot zones** in Chicago
using three algorithms: **K-Means**, **DBSCAN**, and **Hierarchical Clustering**.
""")

tab1, tab2, tab3 = st.tabs(
    ["K-Means Interactive", "Algorithm Comparison", "Cluster Profiles"])

# ── Tab 1: Interactive K-Means ────────────────────────────
with tab1:
    st.subheader("Interactive K-Means Geographic Clustering")

    col1, col2 = st.columns([1, 3])
    with col1:
        k = st.slider("Number of clusters (k)", 2, 10, 5)
        sample_n = st.select_slider("Sample size",
            options=[5000, 10000, 20000, 50000], value=10000)
        run_btn = st.button("▶ Run K-Means", type="primary")

    if run_btn or "geo_labels" not in st.session_state:
        sample_idx = X.sample(sample_n, random_state=42).index
        Xg = X.loc[sample_idx, ["Lat_Scaled","Lon_Scaled"]].values
        mg = meta.loc[sample_idx].copy()

        with st.spinner(f"Running K-Means with k={k}…"):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(Xg)
            mg["Cluster"] = km.labels_.astype(str)
            sil = silhouette_score(Xg, km.labels_,
                                   sample_size=2000, random_state=42)
            db  = davies_bouldin_score(Xg, km.labels_)

        st.session_state["geo_labels"] = mg
        st.session_state["sil"] = sil
        st.session_state["db"]  = db

    mg    = st.session_state.get("geo_labels", None)
    sil   = st.session_state.get("sil", 0)
    db    = st.session_state.get("db",  0)

    if mg is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Silhouette Score", f"{sil:.4f}",
                  delta="≥0.5 is good" if sil >= 0.5 else "< 0.5")
        c2.metric("Davies-Bouldin Index", f"{db:.4f}",
                  delta="lower = better")
        c3.metric("Clusters Found", k)

        fig = px.scatter_mapbox(
            mg.dropna(subset=["Latitude","Longitude"]),
            lat="Latitude", lon="Longitude",
            color="Cluster",
            hover_data=["Primary Type","Crime_Severity_Score","Arrest"],
            zoom=10, height=580,
            title=f"K-Means Geographic Clusters (k={k})",
            mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Algorithm Comparison ───────────────────────────
with tab2:
    st.subheader("Algorithm Comparison — Silhouette Scores")
    st.image("data/cluster_charts/geo_comparison.png",
             caption="Silhouette Score Comparison across 3 algorithms",
             use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### K-Means")
        st.image("data/cluster_charts/geo_kmeans.png",
                 use_container_width=True)
        st.caption("Circular hotspot zones with clear centers")
    with col2:
        st.markdown("### DBSCAN")
        st.image("data/cluster_charts/geo_dbscan.png",
                 use_container_width=True)
        st.caption("Density-based: filters out isolated incidents")
    with col3:
        st.markdown("### Hierarchical")
        st.image("data/cluster_charts/geo_hierarchical.png",
                 use_container_width=True)
        st.caption("Nested zone relationships")

    st.image("data/cluster_charts/kmeans_elbow.png",
             caption="Elbow Method & Silhouette Score vs k",
             use_container_width=True)

    compare_data = {
        "Algorithm"           : ["K-Means","DBSCAN","Hierarchical"],
        "Silhouette Score"    : [0.487, "N/A*", 0.460],
        "Davies-Bouldin"      : [0.773, "N/A*", 0.772],
        "Best For"            : [
            "Patrol zone planning",
            "Dense hotspot detection",
            "Nested zone analysis"],
    }
    st.table(pd.DataFrame(compare_data))
    st.caption("*DBSCAN merged all points into 1 cluster at this eps — "
               "tuning eps lower would reveal more zones.")

# ── Tab 3: Cluster Profiles ───────────────────────────────
with tab3:
    st.subheader("Cluster Profiles from Best Model (K-Means)")
    if "Geo_Cluster" in geo_result.columns:
        profile = geo_result.groupby("Geo_Cluster").agg(
            Total_Crimes  =("Primary Type","count"),
            Top_Crime     =("Primary Type", lambda x: x.value_counts().idxmax()),
            Arrest_Rate   =("Arrest",       lambda x: x.mean()*100),
            Avg_Severity  =("Crime_Severity_Score","mean"),
        ).round(2).reset_index()
        profile.columns = ["Cluster","Total Crimes","Top Crime",
                           "Arrest Rate (%)","Avg Severity"]
        st.dataframe(profile, use_container_width=True, hide_index=True)

        fig = px.bar(profile, x="Cluster", y="Total Crimes",
                     color="Avg Severity",
                     color_continuous_scale="RdYlGn_r",
                     title="Crime Count & Severity by Geographic Cluster")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Cluster result file not found — run Step 4 first.")
