"""
=============================================================
PatrolIQ - Step 4: Unsupervised Learning — Clustering
=============================================================
Geographic Crime Hotspot Clustering (3 algorithms):
  1. K-Means         — circular hotspot zones
  2. DBSCAN          — density-based spatial clusters
  3. Hierarchical    — nested zone relationships

Temporal Pattern Clustering:
  4. K-Means on hour/day/month features

Evaluation metrics:
  - Silhouette Score (target > 0.5)
  - Davies-Bouldin Index (lower = better)
  - Calinski-Harabasz Index (higher = better)

All experiments tracked with MLflow.
Best geographic model saved to disk.
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)
from sklearn.preprocessing import StandardScaler
import mlflow, mlflow.sklearn
import pickle, os, warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────
FEATURES_PATH = "data/features.csv"
META_PATH     = "data/metadata.csv"
CHARTS_DIR    = "data/cluster_charts"
MODELS_DIR    = "models"
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# We subsample for clustering (full 494K is slow for hierarchical/silhouette)
GEO_SAMPLE   = 50_000
TEMP_SAMPLE  = 50_000
RANDOM_STATE = 42
# ──────────────────────────────────────────────────────────


def load_data():
    X    = pd.read_csv(FEATURES_PATH)
    meta = pd.read_csv(META_PATH, parse_dates=["Date"])
    print(f"Feature matrix : {X.shape}")
    print(f"Metadata       : {meta.shape}")
    return X, meta


# ─────────────────────────────────────────────────────────
# HELPER: evaluate a fitted cluster label array
# ─────────────────────────────────────────────────────────
def evaluate(X_arr: np.ndarray, labels: np.ndarray,
             name: str) -> dict:
    """Compute clustering quality metrics."""
    # Filter out noise label -1 (DBSCAN)
    mask = labels != -1
    n_clusters = len(set(labels[mask]))

    if n_clusters < 2:
        print(f"  [{name}] Only {n_clusters} cluster(s) — cannot evaluate")
        return {"silhouette": -1, "davies_bouldin": 99,
                "calinski_harabasz": 0, "n_clusters": n_clusters,
                "noise_pct": (~mask).mean() * 100}

    sil = silhouette_score(X_arr[mask], labels[mask],
                           sample_size=min(5000, mask.sum()),
                           random_state=RANDOM_STATE)
    db  = davies_bouldin_score(X_arr[mask], labels[mask])
    ch  = calinski_harabasz_score(X_arr[mask], labels[mask])
    noise_pct = (~mask).mean() * 100

    print(f"  [{name}] Clusters={n_clusters} | "
          f"Silhouette={sil:.4f} | DB={db:.4f} | CH={ch:.1f} | "
          f"Noise={noise_pct:.1f}%")
    return {"silhouette": sil, "davies_bouldin": db,
            "calinski_harabasz": ch, "n_clusters": n_clusters,
            "noise_pct": noise_pct}


# ─────────────────────────────────────────────────────────
# PLOT HELPER: scatter by cluster label
# ─────────────────────────────────────────────────────────
def scatter_clusters(lon: np.ndarray, lat: np.ndarray,
                     labels: np.ndarray, title: str,
                     filepath: str) -> None:
    unique_labels = sorted(set(labels))
    colors = cm.tab20(np.linspace(0, 1, len(unique_labels)))
    color_map = {lbl: colors[i] for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(9, 10))
    for lbl in unique_labels:
        mask = labels == lbl
        label_name = f"Noise ({mask.sum():,})" if lbl == -1 \
                     else f"Cluster {lbl} ({mask.sum():,})"
        ax.scatter(lon[mask], lat[mask],
                   c=[color_map[lbl]], s=1, alpha=0.3,
                   label=label_name)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    # Only show legend if not too many clusters
    if len(unique_labels) <= 15:
        ax.legend(loc="upper right", markerscale=5, fontsize=7)
    plt.tight_layout()
    fig.savefig(filepath, dpi=120)
    plt.close(fig)
    print(f"  Chart saved → {filepath}")


# ═══════════════════════════════════════════════════════════
# SECTION A: GEOGRAPHIC CLUSTERING
# ═══════════════════════════════════════════════════════════

def run_geographic_clustering(X: pd.DataFrame,
                               meta: pd.DataFrame) -> dict:
    print("\n" + "=" * 55)
    print("  GEOGRAPHIC CLUSTERING — Lat/Lon features")
    print("=" * 55)

    # Use only geographic coordinates for spatial clustering
    geo_cols = ["Lat_Scaled", "Lon_Scaled"]
    sample_idx = X.sample(GEO_SAMPLE, random_state=RANDOM_STATE).index
    X_geo      = X.loc[sample_idx, geo_cols].values
    meta_geo   = meta.loc[sample_idx]
    lon_raw    = X.loc[sample_idx, "Lon_Scaled"].values
    lat_raw    = X.loc[sample_idx, "Lat_Scaled"].values

    results = {}
    mlflow.set_experiment("PatrolIQ_Geographic_Clustering")

    # ── A1. K-Means Elbow Search ──────────────────────────
    print("\n── A1. K-Means (elbow search k=2..10) ──────────")
    inertias, sil_scores = [], []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_geo)
        inertias.append(km.inertia_)
        ss = silhouette_score(X_geo, km.labels_,
                              sample_size=5000, random_state=RANDOM_STATE)
        sil_scores.append(ss)
        print(f"    k={k} | Inertia={km.inertia_:.1f} | Silhouette={ss:.4f}")

    best_k = range(2, 11)[int(np.argmax(sil_scores))]
    print(f"  → Best k by silhouette: {best_k}")

    # Elbow plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(range(2, 11), inertias, "bo-")
    ax1.set_title("Elbow Method — Inertia")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax2.plot(range(2, 11), sil_scores, "rs-")
    ax2.set_title("Silhouette Score vs k")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette Score")
    plt.tight_layout()
    elbow_path = f"{CHARTS_DIR}/kmeans_elbow.png"
    fig.savefig(elbow_path, dpi=120)
    plt.close(fig)

    # Fit best K-Means
    km_best = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    km_best.fit(X_geo)
    metrics_km = evaluate(X_geo, km_best.labels_, f"K-Means k={best_k}")

    with mlflow.start_run(run_name=f"KMeans_geo_k{best_k}"):
        mlflow.log_param("algorithm",        "KMeans")
        mlflow.log_param("k",                best_k)
        mlflow.log_param("sample_size",      GEO_SAMPLE)
        mlflow.log_metrics(metrics_km)
        mlflow.log_artifact(elbow_path)
        mlflow.sklearn.log_model(km_best, "kmeans_model")

    scatter_clusters(lon_raw, lat_raw, km_best.labels_,
                     f"K-Means Geographic Clusters (k={best_k})",
                     f"{CHARTS_DIR}/geo_kmeans.png")
    results["KMeans"] = {"model": km_best, "labels": km_best.labels_,
                          "metrics": metrics_km}

    # ── A2. DBSCAN ───────────────────────────────────────
    print("\n── A2. DBSCAN ─────────────────────────────────")
    # eps tuned for normalized coordinate space (~0.01 ≈ 1 km in Chicago)
    eps_val, min_samples = 0.015, 30
    db = DBSCAN(eps=eps_val, min_samples=min_samples, n_jobs=-1)
    db.fit(X_geo)
    metrics_db = evaluate(X_geo, db.labels_, "DBSCAN")

    with mlflow.start_run(run_name="DBSCAN_geo"):
        mlflow.log_param("algorithm",   "DBSCAN")
        mlflow.log_param("eps",         eps_val)
        mlflow.log_param("min_samples", min_samples)
        mlflow.log_param("sample_size", GEO_SAMPLE)
        mlflow.log_metrics(metrics_db)

    scatter_clusters(lon_raw, lat_raw, db.labels_,
                     f"DBSCAN Geographic Clusters\n"
                     f"(eps={eps_val}, min_samples={min_samples})",
                     f"{CHARTS_DIR}/geo_dbscan.png")
    results["DBSCAN"] = {"model": db, "labels": db.labels_,
                          "metrics": metrics_db}

    # ── A3. Hierarchical Clustering ───────────────────────
    print("\n── A3. Hierarchical Clustering ────────────────")
    n_hc = best_k   # use same k for fair comparison
    hc = AgglomerativeClustering(n_clusters=n_hc, linkage="ward")
    hc.fit(X_geo)
    metrics_hc = evaluate(X_geo, hc.labels_, f"Hierarchical k={n_hc}")

    with mlflow.start_run(run_name=f"Hierarchical_geo_k{n_hc}"):
        mlflow.log_param("algorithm",   "Hierarchical")
        mlflow.log_param("linkage",     "ward")
        mlflow.log_param("n_clusters",  n_hc)
        mlflow.log_param("sample_size", GEO_SAMPLE)
        mlflow.log_metrics(metrics_hc)

    scatter_clusters(lon_raw, lat_raw, hc.labels_,
                     f"Hierarchical Geographic Clusters (k={n_hc})",
                     f"{CHARTS_DIR}/geo_hierarchical.png")
    results["Hierarchical"] = {"model": hc, "labels": hc.labels_,
                                 "metrics": metrics_hc}

    # ── A4. Compare algorithms ────────────────────────────
    print("\n── Algorithm Comparison ───────────────────────")
    compare_df = pd.DataFrame({
        alg: r["metrics"] for alg, r in results.items()
    }).T
    print(compare_df[["n_clusters", "silhouette",
                       "davies_bouldin", "calinski_harabasz",
                       "noise_pct"]].to_string())

    # Select best by silhouette score
    best_alg = compare_df["silhouette"].idxmax()
    print(f"\n  🏆 Best algorithm: {best_alg}  "
          f"(silhouette={compare_df.loc[best_alg,'silhouette']:.4f})")

    # Attach cluster labels to full metadata for export
    # (labels are only for the sample; we'll store them)
    label_col = f"Geo_Cluster_{best_alg}"
    meta_geo   = meta_geo.copy()
    meta_geo["Geo_Cluster"] = results[best_alg]["labels"]
    meta_geo.to_csv("data/geo_cluster_results.csv", index=False)

    # Save best model
    best_model = results[best_alg]["model"]
    with open(f"{MODELS_DIR}/best_geo_model.pkl", "wb") as f:
        pickle.dump({"algorithm": best_alg,
                     "model": best_model,
                     "sample_idx": sample_idx.tolist()}, f)
    print(f"  Best model saved → {MODELS_DIR}/best_geo_model.pkl")

    return results


# ═══════════════════════════════════════════════════════════
# SECTION B: TEMPORAL CLUSTERING
# ═══════════════════════════════════════════════════════════

def run_temporal_clustering(X: pd.DataFrame,
                             meta: pd.DataFrame) -> None:
    print("\n" + "=" * 55)
    print("  TEMPORAL CLUSTERING — Hour / Day / Month")
    print("=" * 55)

    temp_cols  = ["Hour", "DayOfWeek_Num", "Month", "Season_Num",
                  "Is_Weekend_Num"]
    sample_idx = X.sample(TEMP_SAMPLE, random_state=RANDOM_STATE).index
    X_temp     = X.loc[sample_idx, temp_cols].values
    meta_temp  = meta.loc[sample_idx].copy()

    # Standardize so hour/month are on the same scale
    scaler = StandardScaler()
    X_temp_scaled = scaler.fit_transform(X_temp)

    mlflow.set_experiment("PatrolIQ_Temporal_Clustering")

    print("\n── K-Means temporal (k=3..6) ──────────────────")
    best_k, best_sil, best_labels = 3, -1, None
    for k in range(3, 7):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_temp_scaled)
        ss = silhouette_score(X_temp_scaled, km.labels_,
                              sample_size=5000, random_state=RANDOM_STATE)
        print(f"    k={k} | Silhouette={ss:.4f}")
        if ss > best_sil:
            best_sil, best_k, best_labels = ss, k, km.labels_

        with mlflow.start_run(run_name=f"KMeans_temporal_k{k}"):
            mlflow.log_param("algorithm",   "KMeans_temporal")
            mlflow.log_param("k",           k)
            mlflow.log_metrics({"silhouette": ss})

    print(f"\n  → Best temporal k={best_k} (silhouette={best_sil:.4f})")

    # Analyse each temporal cluster
    meta_temp["Temporal_Cluster"] = best_labels
    meta_temp["Hour"] = X.loc[sample_idx, "Hour"].values

    print("\n── Temporal Cluster Profiles ──────────────────")
    for c in sorted(meta_temp["Temporal_Cluster"].unique()):
        grp = meta_temp[meta_temp["Temporal_Cluster"] == c]
        avg_hour = X.loc[grp.index, "Hour"].mean()
        top_crime = grp["Primary Type"].value_counts().idxmax()
        print(f"  Cluster {c} | n={len(grp):,} | "
              f"Avg hour={avg_hour:.1f} | Top crime={top_crime}")

    # Hourly heatmap: crimes per hour per cluster
    hour_vals = X.loc[sample_idx, "Hour"].values
    fig, ax = plt.subplots(figsize=(13, 5))
    for c in sorted(set(best_labels)):
        mask = best_labels == c
        hour_counts = pd.Series(hour_vals[mask]).value_counts().sort_index()
        ax.plot(hour_counts.index, hour_counts.values,
                marker="o", label=f"Cluster {c}", linewidth=2)
    ax.set_title(f"Temporal Clusters — Crime Count by Hour (k={best_k})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour of Day (0–23)")
    ax.set_ylabel("Number of Crimes")
    ax.legend()
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    path = f"{CHARTS_DIR}/temporal_kmeans.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Chart saved → {path}")

    meta_temp.to_csv("data/temporal_cluster_results.csv", index=False)


# ─── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    X, meta = load_data()
    geo_results  = run_geographic_clustering(X, meta)
    run_temporal_clustering(X, meta)
    print("\n✅ Step 4 Complete — Clustering done, results & models saved")
