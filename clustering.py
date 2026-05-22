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

Evaluation: Silhouette, Davies-Bouldin, Calinski-Harabasz
All experiments tracked with MLflow.
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

mlflow.set_tracking_uri("sqlite:///mlflow.db")

import pickle, os, warnings
warnings.filterwarnings("ignore")

FEATURES_PATH = "data/features.csv"
META_PATH     = "data/metadata.csv"
CHARTS_DIR    = "data/cluster_charts"
MODELS_DIR    = "models"
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

GEO_SAMPLE   = 20_000   # kept small so DBSCAN/Hierarchical stay in memory
TEMP_SAMPLE  = 20_000
RANDOM_STATE = 42


def load_data():
    X    = pd.read_csv(r"C:\Users\bhuva\OneDrive\Desktop\new desktop\desktop\DS PORTAL\patrol coding files\data\features.csv")
    meta = pd.read_csv(r"C:\Users\bhuva\OneDrive\Desktop\new desktop\desktop\DS PORTAL\patrol coding files\data\metadata.csv", parse_dates=["Date"])
    print(f"Feature matrix : {X.shape}")
    return X, meta


def evaluate(X_arr, labels, name):
    mask      = labels != -1
    n_clusters = len(set(labels[mask]))
    if n_clusters < 2:
        print(f"  [{name}] Only {n_clusters} cluster — skipping metrics")
        return {"silhouette": -1, "davies_bouldin": 99,
                "calinski_harabasz": 0, "n_clusters": n_clusters,
                "noise_pct": float((~mask).mean() * 100)}
    sil = silhouette_score(X_arr[mask], labels[mask],
                           sample_size=min(3000, mask.sum()),
                           random_state=RANDOM_STATE)
    db  = davies_bouldin_score(X_arr[mask], labels[mask])
    ch  = calinski_harabasz_score(X_arr[mask], labels[mask])
    noise_pct = float((~mask).mean() * 100)
    print(f"  [{name}] k={n_clusters} | Sil={sil:.4f} | "
          f"DB={db:.4f} | CH={ch:.1f} | Noise={noise_pct:.1f}%")
    return {"silhouette": float(sil), "davies_bouldin": float(db),
            "calinski_harabasz": float(ch), "n_clusters": int(n_clusters),
            "noise_pct": noise_pct}


def scatter_clusters(lon, lat, labels, title, filepath):
    unique = sorted(set(labels))
    colors = cm.tab20(np.linspace(0, 1, max(len(unique), 2)))
    cmap   = {lbl: colors[i] for i, lbl in enumerate(unique)}
    fig, ax = plt.subplots(figsize=(9, 10))
    for lbl in unique:
        m = labels == lbl
        tag = f"Noise ({m.sum():,})" if lbl == -1 else f"Cluster {lbl} ({m.sum():,})"
        ax.scatter(lon[m], lat[m], c=[cmap[lbl]], s=1.5,
                   alpha=0.35, label=tag)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude (scaled)")
    ax.set_ylabel("Latitude (scaled)")
    if len(unique) <= 15:
        ax.legend(loc="upper right", markerscale=6, fontsize=7)
    plt.tight_layout()
    fig.savefig(filepath, dpi=120); plt.close(fig)
    print(f"  Chart → {filepath}")


# ══════════════════════════════════════════════════
# A — GEOGRAPHIC CLUSTERING
# ══════════════════════════════════════════════════
def run_geographic_clustering(X, meta):
    print("\n" + "="*52)
    print("  GEOGRAPHIC CLUSTERING")
    print("="*52)

    geo_cols   = ["Lat_Scaled", "Lon_Scaled"]
    sample_idx = X.sample(GEO_SAMPLE, random_state=RANDOM_STATE).index
    X_geo      = X.loc[sample_idx, geo_cols].values
    meta_geo   = meta.loc[sample_idx].copy()
    lon_s      = X_geo[:, 1]
    lat_s      = X_geo[:, 0]

    results = {}
    mlflow.set_experiment("PatrolIQ_Geographic_Clustering")

    # ── A1 K-Means elbow ──────────────────────────
    print("\n── A1. K-Means (k=2..10) ──────────────────────")
    inertias, sils = [], []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_geo)
        inertias.append(km.inertia_)
        ss = silhouette_score(X_geo, km.labels_,
                              sample_size=3000, random_state=RANDOM_STATE)
        sils.append(ss)
        print(f"    k={k} | Inertia={km.inertia_:.2f} | Sil={ss:.4f}")

    best_k = range(2, 11)[int(np.argmax(sils))]
    print(f"  → Best k={best_k}")

    # Elbow plot
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(range(2,11), inertias, "bo-"); a1.set_title("Elbow — Inertia")
    a1.set_xlabel("k"); a1.set_ylabel("Inertia")
    a2.plot(range(2,11), sils, "rs-"); a2.set_title("Silhouette vs k")
    a2.set_xlabel("k"); a2.set_ylabel("Silhouette")
    plt.tight_layout()
    ep = f"{CHARTS_DIR}/kmeans_elbow.png"
    fig.savefig(ep, dpi=120); plt.close(fig)

    km_best = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    km_best.fit(X_geo)
    m_km = evaluate(X_geo, km_best.labels_, f"K-Means k={best_k}")

    with mlflow.start_run(run_name=f"KMeans_geo_k{best_k}"):
        mlflow.log_param("algorithm", "KMeans")
        mlflow.log_param("k", best_k)
        mlflow.log_param("sample_size", GEO_SAMPLE)
        mlflow.log_metrics(m_km)
        mlflow.log_artifact(ep)
        mlflow.sklearn.log_model(km_best, "model")

    scatter_clusters(lon_s, lat_s, km_best.labels_,
                     f"K-Means Geographic Clusters (k={best_k})",
                     f"{CHARTS_DIR}/geo_kmeans.png")
    results["KMeans"] = {"model": km_best,
                         "labels": km_best.labels_, "metrics": m_km}

    # ── A2 DBSCAN ─────────────────────────────────
    print("\n── A2. DBSCAN ─────────────────────────────────")
    eps_val, min_s = 0.02, 20
    db = DBSCAN(eps=eps_val, min_samples=min_s, algorithm="ball_tree")
    db.fit(X_geo)
    m_db = evaluate(X_geo, db.labels_, "DBSCAN")

    with mlflow.start_run(run_name="DBSCAN_geo"):
        mlflow.log_param("algorithm",   "DBSCAN")
        mlflow.log_param("eps",         eps_val)
        mlflow.log_param("min_samples", min_s)
        mlflow.log_param("sample_size", GEO_SAMPLE)
        mlflow.log_metrics(m_db)

    scatter_clusters(lon_s, lat_s, db.labels_,
                     f"DBSCAN Geographic Clusters\n(eps={eps_val}, min_samples={min_s})",
                     f"{CHARTS_DIR}/geo_dbscan.png")
    results["DBSCAN"] = {"model": db, "labels": db.labels_, "metrics": m_db}

    # ── A3 Hierarchical ───────────────────────────
    print("\n── A3. Hierarchical Clustering ────────────────")
    hc = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    hc.fit(X_geo)
    m_hc = evaluate(X_geo, hc.labels_, f"Hierarchical k={best_k}")

    with mlflow.start_run(run_name=f"Hierarchical_geo_k{best_k}"):
        mlflow.log_param("algorithm",  "Hierarchical")
        mlflow.log_param("linkage",    "ward")
        mlflow.log_param("n_clusters", best_k)
        mlflow.log_param("sample_size", GEO_SAMPLE)
        mlflow.log_metrics(m_hc)

    scatter_clusters(lon_s, lat_s, hc.labels_,
                     f"Hierarchical Geographic Clusters (k={best_k})",
                     f"{CHARTS_DIR}/geo_hierarchical.png")
    results["Hierarchical"] = {"model": hc,
                                "labels": hc.labels_, "metrics": m_hc}

    # ── Compare ───────────────────────────────────
    print("\n── Algorithm Comparison ───────────────────────")
    cmp = pd.DataFrame({a: r["metrics"] for a, r in results.items()}).T
    print(cmp[["n_clusters","silhouette","davies_bouldin",
               "calinski_harabasz","noise_pct"]].to_string())
    best_alg = cmp["silhouette"].idxmax()
    print(f"\n  🏆 Best: {best_alg}  "
          f"(Silhouette={cmp.loc[best_alg,'silhouette']:.4f})")

    # Comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    algs = list(results.keys())
    sil_vals = [results[a]["metrics"]["silhouette"] for a in algs]
    bars = ax.bar(algs, sil_vals, color=["steelblue","tomato","seagreen"])
    ax.axhline(0.5, color="red", linestyle="--", label="Target (0.5)")
    ax.set_title("Geographic Clustering — Silhouette Score Comparison",
                 fontweight="bold")
    ax.set_ylabel("Silhouette Score")
    ax.legend()
    for bar, v in zip(bars, sil_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(f"{CHARTS_DIR}/geo_comparison.png", dpi=120)
    plt.close(fig)

    meta_geo["Geo_Cluster"] = results[best_alg]["labels"]
    meta_geo.to_csv("data/geo_cluster_results.csv", index=False)
    with open(f"{MODELS_DIR}/best_geo_model.pkl", "wb") as f:
        pickle.dump({"algorithm": best_alg,
                     "model": results[best_alg]["model"],
                     "sample_idx": sample_idx.tolist()}, f)
    print(f"  Best model saved → {MODELS_DIR}/best_geo_model.pkl")
    return results


# ══════════════════════════════════════════════════
# B — TEMPORAL CLUSTERING
# ══════════════════════════════════════════════════
def run_temporal_clustering(X, meta):
    print("\n" + "="*52)
    print("  TEMPORAL CLUSTERING")
    print("="*52)

    temp_cols  = ["Hour", "DayOfWeek_Num", "Month",
                  "Season_Num", "Is_Weekend_Num"]
    sample_idx = X.sample(TEMP_SAMPLE, random_state=RANDOM_STATE).index
    X_t        = StandardScaler().fit_transform(
                     X.loc[sample_idx, temp_cols].values)
    meta_t     = meta.loc[sample_idx].copy()
    hour_vals  = X.loc[sample_idx, "Hour"].values

    mlflow.set_experiment("PatrolIQ_Temporal_Clustering")

    print("\n── K-Means temporal (k=3..6) ──────────────────")
    best_k, best_sil, best_labels = 3, -1, None
    for k in range(3, 7):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_t)
        ss = silhouette_score(X_t, km.labels_,
                              sample_size=3000, random_state=RANDOM_STATE)
        print(f"    k={k} | Silhouette={ss:.4f}")
        if ss > best_sil:
            best_sil, best_k, best_labels = ss, k, km.labels_.copy()
        with mlflow.start_run(run_name=f"KMeans_temporal_k{k}"):
            mlflow.log_param("algorithm", "KMeans_temporal")
            mlflow.log_param("k", k)
            mlflow.log_metrics({"silhouette": float(ss)})

    print(f"\n  → Best temporal k={best_k} (Sil={best_sil:.4f})")

    meta_t["Temporal_Cluster"] = best_labels
    meta_t["Hour_val"]         = hour_vals

    print("\n── Temporal Cluster Profiles ──────────────────")
    for c in sorted(set(best_labels)):
        grp  = meta_t[meta_t["Temporal_Cluster"] == c]
        havg = grp["Hour_val"].mean()
        top  = grp["Primary Type"].value_counts().idxmax()
        print(f"  Cluster {c} | n={len(grp):,} | "
              f"Avg hour={havg:.1f} | Top crime: {top}")

    # Hourly line chart
    fig, ax = plt.subplots(figsize=(13, 5))
    for c in sorted(set(best_labels)):
        m = best_labels == c
        hc = pd.Series(hour_vals[m]).value_counts().sort_index()
        ax.plot(hc.index, hc.values, marker="o",
                label=f"Cluster {c}", linewidth=2)
    ax.set_title(f"Temporal Clusters — Crime Count by Hour (k={best_k})",
                 fontweight="bold")
    ax.set_xlabel("Hour (0–23)"); ax.set_ylabel("Count")
    ax.set_xticks(range(24)); ax.legend()
    plt.tight_layout()
    fig.savefig(f"{CHARTS_DIR}/temporal_kmeans.png", dpi=120)
    plt.close(fig)
    print(f"  Chart → {CHARTS_DIR}/temporal_kmeans.png")

    meta_t.to_csv("data/temporal_cluster_results.csv", index=False)


# ─── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    X, meta = load_data()
    run_geographic_clustering(X, meta)
    run_temporal_clustering(X, meta)
    print("\n✅ Step 4 Complete — Clustering done & logged to MLflow")
