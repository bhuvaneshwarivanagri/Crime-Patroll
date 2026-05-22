"""
=============================================================
PatrolIQ - Step 5: Dimensionality Reduction
=============================================================
Techniques:
  1. PCA  — reduce 19 features → 2-3 components (target ≥70% variance)
             identify top-5 features driving crime patterns
  2. t-SNE — 2D visualization of high-dimensional clusters
  3. UMAP  — faster alternative to t-SNE

All experiments logged to MLflow.
Outputs saved to data/ and data/dim_reduction_charts/
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
import os, warnings
warnings.filterwarnings("ignore")

FEATURES_PATH = "data/features.csv"
META_PATH     = "data/metadata.csv"
CHARTS_DIR    = "data/dim_reduction_charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

SAMPLE       = 8_000   # t-SNE is O(n²) — keep manageable
RANDOM_STATE = 42


def load_and_scale():
    X    = pd.read_csv(r"C:\Users\bhuva\OneDrive\Desktop\new desktop\desktop\DS PORTAL\patrol coding files\data\features.csv")
    meta = pd.read_csv(r"C:\Users\bhuva\OneDrive\Desktop\new desktop\desktop\DS PORTAL\patrol coding files\data\metadata.csv", parse_dates=["Date"])

    idx    = X.sample(SAMPLE, random_state=RANDOM_STATE).index
    X_s    = X.loc[idx]
    meta_s = meta.loc[idx].copy()

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_s.values)
    feature_names = X_s.columns.tolist()

    print(f"Sample shape : {X_scaled.shape}")
    print(f"Features     : {feature_names}")
    return X_scaled, meta_s, feature_names, idx


# ══════════════════════════════════════════════════
# 1 — PCA
# ══════════════════════════════════════════════════
def run_pca(X_scaled, meta_s, feature_names):
    print("\n" + "="*52)
    print("  PCA — Principal Component Analysis")
    print("="*52)

    mlflow.set_experiment("PatrolIQ_Dimensionality_Reduction")

    # Fit full PCA to get explained variance
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)

    # How many components for ≥70% variance?
    n70 = int(np.argmax(cum_var >= 0.70)) + 1
    n80 = int(np.argmax(cum_var >= 0.80)) + 1
    print(f"  Components for 70% variance : {n70}")
    print(f"  Components for 80% variance : {n80}")

    # Scree plot
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    n_show = min(15, len(cum_var))
    a1.bar(range(1, n_show+1),
           pca_full.explained_variance_ratio_[:n_show] * 100,
           color="steelblue")
    a1.set_title("Scree Plot — Explained Variance per Component")
    a1.set_xlabel("Principal Component"); a1.set_ylabel("Variance Explained (%)")

    a2.plot(range(1, n_show+1), cum_var[:n_show] * 100, "go-")
    a2.axhline(70, color="red",    ls="--", label="70% threshold")
    a2.axhline(80, color="orange", ls="--", label="80% threshold")
    a2.set_title("Cumulative Explained Variance")
    a2.set_xlabel("Number of Components"); a2.set_ylabel("Cumulative Variance (%)")
    a2.legend()
    plt.tight_layout()
    scree_path = f"{CHARTS_DIR}/pca_scree.png"
    fig.savefig(scree_path, dpi=120); plt.close(fig)
    print(f"  Chart → {scree_path}")

    # Feature importance from first 3 PCs (loadings)
    pca3 = PCA(n_components=3, random_state=RANDOM_STATE)
    X_pca3 = pca3.fit_transform(X_scaled)

    print(f"\n  PC1 variance: {pca3.explained_variance_ratio_[0]*100:.2f}%")
    print(f"  PC2 variance: {pca3.explained_variance_ratio_[1]*100:.2f}%")
    print(f"  PC3 variance: {pca3.explained_variance_ratio_[2]*100:.2f}%")
    print(f"  Total (3 PCs): {pca3.explained_variance_ratio_.sum()*100:.2f}%")

    # Top-5 features per component (absolute loading)
    loadings = pd.DataFrame(pca3.components_.T,
                            index=feature_names,
                            columns=["PC1", "PC2", "PC3"])
    print("\n── Top-5 features by |loading| ─────────────────")
    for pc in ["PC1", "PC2", "PC3"]:
        top5 = loadings[pc].abs().sort_values(ascending=False).head(5)
        print(f"  {pc}: " + ", ".join(
            [f"{f}({v:.3f})" for f, v in top5.items()]))

    # Feature importance heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(loadings.values.T, cmap="RdBu_r", aspect="auto",
                   vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["PC1", "PC2", "PC3"])
    ax.set_title("PCA Feature Loadings (3 Components)", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Loading value")
    plt.tight_layout()
    load_path = f"{CHARTS_DIR}/pca_loadings.png"
    fig.savefig(load_path, dpi=120); plt.close(fig)
    print(f"  Chart → {load_path}")

    # 2D scatter colored by crime severity
    severity = meta_s["Crime_Severity_Score"].values
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(X_pca3[:, 0], X_pca3[:, 1],
                    c=severity, cmap="RdYlGn_r", alpha=0.4, s=5)
    plt.colorbar(sc, ax=ax, label="Crime Severity Score")
    ax.set_title("PCA 2D Projection — Colored by Crime Severity",
                 fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca3.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca3.explained_variance_ratio_[1]*100:.1f}%)")
    plt.tight_layout()
    scatter2d_path = f"{CHARTS_DIR}/pca_scatter2d.png"
    fig.savefig(scatter2d_path, dpi=120); plt.close(fig)
    print(f"  Chart → {scatter2d_path}")

    # MLflow logging
    with mlflow.start_run(run_name="PCA_3components"):
        mlflow.log_param("technique",    "PCA")
        mlflow.log_param("n_components", 3)
        mlflow.log_param("sample_size",  SAMPLE)
        mlflow.log_metric("pc1_variance_pct",
                          float(pca3.explained_variance_ratio_[0]*100))
        mlflow.log_metric("pc2_variance_pct",
                          float(pca3.explained_variance_ratio_[1]*100))
        mlflow.log_metric("pc3_variance_pct",
                          float(pca3.explained_variance_ratio_[2]*100))
        mlflow.log_metric("total_variance_pct",
                          float(pca3.explained_variance_ratio_.sum()*100))
        mlflow.log_metric("components_for_70pct_variance", float(n70))
        mlflow.log_artifact(scree_path)
        mlflow.log_artifact(load_path)
        mlflow.log_artifact(scatter2d_path)

    # Save PCA result
    pca_df = pd.DataFrame(X_pca3, columns=["PC1","PC2","PC3"],
                          index=meta_s.index)
    pca_df["Primary_Type"]         = meta_s["Primary Type"].values
    pca_df["Crime_Severity_Score"] = meta_s["Crime_Severity_Score"].values
    pca_df["Season"]               = meta_s["Season"].values
    pca_df.to_csv("data/pca_results.csv", index=False)

    return X_pca3, pca3, loadings


# ══════════════════════════════════════════════════
# 2 — t-SNE
# ══════════════════════════════════════════════════
def run_tsne(X_scaled, meta_s):
    print("\n" + "="*52)
    print("  t-SNE — 2D Visualization")
    print("="*52)

    # Use top 10 PCA components as input to t-SNE for speed
    pca10   = PCA(n_components=10, random_state=RANDOM_STATE)
    X_pca10 = pca10.fit_transform(X_scaled)
    print(f"  Pre-reduced to 10 PCs "
          f"({pca10.explained_variance_ratio_.sum()*100:.1f}% variance) "
          f"before t-SNE")

    tsne = TSNE(n_components=2, perplexity=40, learning_rate=200,
                max_iter=1000, random_state=RANDOM_STATE, n_jobs=1)
    X_tsne = tsne.fit_transform(X_pca10)
    print(f"  t-SNE KL divergence: {tsne.kl_divergence_:.4f}")

    # ── Plot 1: colored by crime type (top 8) ─────
    top_crimes = meta_s["Primary Type"].value_counts().head(8).index.tolist()
    colors8    = cm.tab10(np.linspace(0, 1, 8))
    fig, ax = plt.subplots(figsize=(11, 8))
    for i, ct in enumerate(top_crimes):
        m = meta_s["Primary Type"].values == ct
        ax.scatter(X_tsne[m, 0], X_tsne[m, 1],
                   c=[colors8[i]], s=3, alpha=0.4, label=ct)
    others = ~meta_s["Primary Type"].isin(top_crimes).values
    ax.scatter(X_tsne[others, 0], X_tsne[others, 1],
               c="lightgrey", s=2, alpha=0.2, label="Other")
    ax.set_title("t-SNE 2D — Colored by Crime Type (Top 8)",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel("t-SNE Dim 1"); ax.set_ylabel("t-SNE Dim 2")
    ax.legend(loc="upper right", markerscale=4, fontsize=8)
    plt.tight_layout()
    crime_path = f"{CHARTS_DIR}/tsne_by_crime_type.png"
    fig.savefig(crime_path, dpi=120); plt.close(fig)
    print(f"  Chart → {crime_path}")

    # ── Plot 2: colored by hour (day vs night) ────
    hours  = meta_s["Date"].dt.hour.values
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                    c=hours, cmap="twilight", alpha=0.4, s=3)
    plt.colorbar(sc, ax=ax, label="Hour of Day")
    ax.set_title("t-SNE 2D — Colored by Hour of Day",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel("t-SNE Dim 1"); ax.set_ylabel("t-SNE Dim 2")
    plt.tight_layout()
    hour_path = f"{CHARTS_DIR}/tsne_by_hour.png"
    fig.savefig(hour_path, dpi=120); plt.close(fig)
    print(f"  Chart → {hour_path}")

    # ── Plot 3: colored by severity ───────────────
    severity = meta_s["Crime_Severity_Score"].values
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                    c=severity, cmap="RdYlGn_r", alpha=0.4, s=3)
    plt.colorbar(sc, ax=ax, label="Severity Score")
    ax.set_title("t-SNE 2D — Colored by Crime Severity",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel("t-SNE Dim 1"); ax.set_ylabel("t-SNE Dim 2")
    plt.tight_layout()
    sev_path = f"{CHARTS_DIR}/tsne_by_severity.png"
    fig.savefig(sev_path, dpi=120); plt.close(fig)
    print(f"  Chart → {sev_path}")

    with mlflow.start_run(run_name="tSNE_2d"):
        mlflow.log_param("technique",   "t-SNE")
        mlflow.log_param("perplexity",  40)
        mlflow.log_param("n_iter",      1000)
        mlflow.log_param("sample_size", SAMPLE)
        mlflow.log_metric("kl_divergence", float(tsne.kl_divergence_))
        mlflow.log_artifact(crime_path)
        mlflow.log_artifact(hour_path)
        mlflow.log_artifact(sev_path)

    # Save t-SNE result
    tsne_df = pd.DataFrame(X_tsne, columns=["TSNE1","TSNE2"])
    tsne_df["Primary_Type"]         = meta_s["Primary Type"].values
    tsne_df["Crime_Severity_Score"] = meta_s["Crime_Severity_Score"].values
    tsne_df["Hour"]                 = hours
    tsne_df.to_csv("data/tsne_results.csv", index=False)

    return X_tsne


# ══════════════════════════════════════════════════
# 3 — UMAP  (optional, falls back gracefully)
# ══════════════════════════════════════════════════
def run_umap(X_scaled, meta_s):
    print("\n" + "="*52)
    print("  UMAP — Uniform Manifold Approximation")
    print("="*52)
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=30,
                            min_dist=0.1, random_state=RANDOM_STATE)
        X_umap = reducer.fit_transform(X_scaled)
        print("  UMAP fit complete")

        fig, ax = plt.subplots(figsize=(10, 8))
        top_crimes = meta_s["Primary Type"].value_counts().head(8).index.tolist()
        colors8    = cm.tab10(np.linspace(0, 1, 8))
        for i, ct in enumerate(top_crimes):
            m = meta_s["Primary Type"].values == ct
            ax.scatter(X_umap[m, 0], X_umap[m, 1],
                       c=[colors8[i]], s=3, alpha=0.4, label=ct)
        ax.set_title("UMAP 2D — Colored by Crime Type",
                     fontweight="bold", fontsize=13)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.legend(loc="upper right", markerscale=4, fontsize=8)
        plt.tight_layout()
        umap_path = f"{CHARTS_DIR}/umap_by_crime_type.png"
        fig.savefig(umap_path, dpi=120); plt.close(fig)
        print(f"  Chart → {umap_path}")

        with mlflow.start_run(run_name="UMAP_2d"):
            mlflow.log_param("technique",   "UMAP")
            mlflow.log_param("n_neighbors", 30)
            mlflow.log_param("min_dist",    0.1)
            mlflow.log_param("sample_size", SAMPLE)
            mlflow.log_artifact(umap_path)

        umap_df = pd.DataFrame(X_umap, columns=["UMAP1","UMAP2"])
        umap_df["Primary_Type"] = meta_s["Primary Type"].values
        umap_df.to_csv("data/umap_results.csv", index=False)
        return X_umap

    except ImportError:
        print("  umap-learn not available — skipping UMAP")
        return None


# ─── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    X_scaled, meta_s, feature_names, idx = load_and_scale()

    mlflow.set_experiment("PatrolIQ_Dimensionality_Reduction")

    X_pca3, pca_model, loadings = run_pca(X_scaled, meta_s, feature_names)
    X_tsne                      = run_tsne(X_scaled, meta_s)
    X_umap                      = run_umap(X_scaled, meta_s)

    print("\n── Summary ─────────────────────────────────────")
    print(f"  PCA  : {X_pca3.shape}  → "
          f"saved to data/pca_results.csv")
    print(f"  t-SNE: {X_tsne.shape}  → "
          f"saved to data/tsne_results.csv")
    if X_umap is not None:
        print(f"  UMAP : {X_umap.shape}  → "
              f"saved to data/umap_results.csv")

    print("\n✅ Step 5 Complete — Dimensionality reduction done")
