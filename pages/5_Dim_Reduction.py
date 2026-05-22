"""Page 5 — Dimensionality Reduction"""
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🔬 Dimensionality Reduction")

@st.cache_data
def load():
    pca  = pd.read_csv("data/pca_results.csv")
    tsne = pd.read_csv("data/tsne_results.csv")
    umap = pd.read_csv("data/umap_results.csv") if \
           __import__("os").path.exists("data/umap_results.csv") else None
    return pca, tsne, umap

pca_df, tsne_df, umap_df = load()

st.markdown("""
Dimensionality reduction compresses 19 features into 2–3 dimensions,
making it possible to **visualize hidden crime patterns** and identify
which features drive crime behavior.
""")

tab1, tab2, tab3 = st.tabs(["PCA", "t-SNE", "UMAP"])

# ── PCA ───────────────────────────────────────────────────
with tab1:
    st.subheader("Principal Component Analysis (PCA)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PC1 Variance", "24.6%")
    col2.metric("PC2 Variance",  "9.7%")
    col3.metric("PC3 Variance",  "9.2%")
    col4.metric("Total (3 PCs)", "43.5%")

    st.info("📌 7 components are needed to explain ≥70% of variance in this dataset. "
            "This is expected — crime data has many independent dimensions "
            "(geography, time, crime type).")

    st.image("data/dim_reduction_charts/pca_scree.png",
             caption="Scree Plot & Cumulative Variance",
             use_container_width=True)

    st.subheader("Feature Importance Heatmap")
    st.image("data/dim_reduction_charts/pca_loadings.png",
             caption="PCA Component Loadings — which features matter most",
             use_container_width=True)

    st.markdown("""
**Key Insights from PCA Loadings:**
- **PC1** (geography axis): Dominated by `Lat_Scaled`, `District`, `Beat`, `Ward`, `Lon_Scaled` — location drives most variance
- **PC2** (time axis): Dominated by `DayOfWeek_Num`, `Is_Weekend_Num` — day-of-week patterns
- **PC3** (grid axis): Dominated by `Grid_ID`, `Lat_Bin` — fine-grained geographic grid cells
""")

    st.subheader("PCA 2D Scatter")
    color_by = st.selectbox("Color by",
        ["Crime_Severity_Score", "Primary_Type", "Season"])
    fig = px.scatter(pca_df, x="PC1", y="PC2",
                     color=color_by,
                     opacity=0.4, size_max=4,
                     title=f"PCA 2D Projection — colored by {color_by}",
                     color_continuous_scale="RdYlGn_r"
                     if color_by == "Crime_Severity_Score" else None)
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ── t-SNE ─────────────────────────────────────────────────
with tab2:
    st.subheader("t-SNE — 2D Non-linear Visualization")
    st.markdown("""
t-SNE preserves **local structure** in the data.
Similar crimes cluster together in 2D space.
""")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("data/dim_reduction_charts/tsne_by_crime_type.png",
                 caption="Colored by Crime Type")
    with col2:
        st.image("data/dim_reduction_charts/tsne_by_hour.png",
                 caption="Colored by Hour of Day")
    with col3:
        st.image("data/dim_reduction_charts/tsne_by_severity.png",
                 caption="Colored by Severity")

    # Interactive t-SNE
    st.subheader("Interactive t-SNE Plot")
    color_tsne = st.selectbox("Color by",
        ["Primary_Type", "Crime_Severity_Score", "Hour"],
        key="tsne_color")
    fig = px.scatter(tsne_df, x="TSNE1", y="TSNE2",
                     color=color_tsne, opacity=0.4,
                     title="t-SNE 2D — Crime Pattern Space",
                     color_continuous_scale="Viridis"
                     if color_tsne != "Primary_Type" else None)
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ── UMAP ──────────────────────────────────────────────────
with tab3:
    st.subheader("UMAP — Uniform Manifold Approximation & Projection")
    st.markdown("""
UMAP preserves both **local and global structure**, runs faster than t-SNE,
and produces more interpretable embeddings.
""")
    if umap_df is not None:
        st.image("data/dim_reduction_charts/umap_by_crime_type.png",
                 caption="UMAP — Colored by Crime Type",
                 use_container_width=True)

        fig = px.scatter(umap_df, x="UMAP1", y="UMAP2",
                         color="Primary_Type", opacity=0.4,
                         title="UMAP 2D Interactive — Crime Pattern Space")
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("UMAP results not found. Install umap-learn and rerun Step 5.")
