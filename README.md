# PatrolIQ — Smart Safety Analytics Platform
> GUVI × HCL Capstone Project | Public Safety & Urban Analytics

---

## 🚀 Quick Start

### 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### 2 — Prepare data
Place `patrol.csv` inside the `data/` folder (500K Chicago crime records).

### 3 — Run all preprocessing & ML steps in order
```bash
python step1_preprocessing.py   # clean data, extract temporal features
python step2_eda.py             # EDA charts
python step3_features.py        # feature engineering & encoding
python step4_clustering.py      # K-Means, DBSCAN, Hierarchical + Temporal
python step5_dim_reduction.py   # PCA, t-SNE, UMAP
```

### 4 — Launch the Streamlit app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

### 5 — Launch MLflow UI (optional)
```bash
mlflow ui --port 5000
```
Open http://localhost:5000 to inspect all logged experiments.

---

## ☁️ Deploy to Streamlit Cloud

1. Push this entire folder to a **GitHub repository**.
2. Go to https://streamlit.io/cloud and click **New app**.
3. Select your repo, branch, and set the main file to `app.py`.
4. Add any secrets if needed in the Streamlit Cloud settings.
5. Click **Deploy** — the app will be live within minutes!

> **Note:** Upload the pre-processed `data/` folder to your repo or use
> `st.cache_data` with the raw CSV on cloud storage (e.g., GitHub LFS
> or Google Drive) for large files.

---

## 🐳 Docker Deployment (Bonus)

```bash
# Build image
docker build -t patroliq .

# Run standalone
docker run -p 8501:8501 -v $(pwd)/data:/app/data patroliq

# Or use docker-compose (includes MLflow server)
docker-compose up -d
```

Streamlit → http://localhost:8501  
MLflow UI  → http://localhost:5000

---

## 📁 Project Structure

```
patroliq/
├── app.py                      # Streamlit main entry point
├── step1_preprocessing.py      # Data cleaning & temporal features
├── step2_eda.py                # Exploratory Data Analysis + charts
├── step3_features.py           # Feature engineering & encoding
├── step4_clustering.py         # Geographic + Temporal clustering
├── step5_dim_reduction.py      # PCA, t-SNE, UMAP
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container definition
├── docker-compose.yml          # Multi-service Docker setup
├── .streamlit/config.toml      # Streamlit theme & server config
├── pages/
│   ├── 1_home.py               # Project overview
│   ├── 2_eda.py                # Interactive EDA dashboard
│   ├── 3_geo_clustering.py     # Geographic clustering page
│   ├── 4_temporal_clustering.py # Temporal patterns page
│   ├── 5_dim_reduction.py      # PCA / t-SNE / UMAP page
│   └── 6_mlflow_tracker.py     # MLflow experiment viewer
├── data/
│   ├── patrol.csv              # Raw input (500K records)
│   ├── cleaned_crimes.csv      # Step 1 output
│   ├── features.csv            # Step 3 feature matrix
│   ├── metadata.csv            # Step 3 metadata
│   ├── geo_cluster_results.csv # Step 4 geographic labels
│   ├── temporal_cluster_results.csv # Step 4 temporal labels
│   ├── pca_results.csv         # Step 5 PCA embeddings
│   ├── tsne_results.csv        # Step 5 t-SNE embeddings
│   ├── umap_results.csv        # Step 5 UMAP embeddings
│   ├── eda_charts/             # Static EDA chart PNGs
│   ├── cluster_charts/         # Clustering chart PNGs
│   └── dim_reduction_charts/   # Dimensionality reduction PNGs
├── models/
│   └── best_geo_model.pkl      # Best geographic clustering model
└── mlruns/                     # MLflow experiment tracking store
```

---

## 📊 Key Results

| Metric                      | Value           |
|-----------------------------|-----------------|
| Total records processed     | 494,423         |
| Crime types                 | 32              |
| Best geo clustering algo    | K-Means (k=2)   |
| Best silhouette score       | 0.487           |
| PCA variance (3 components) | 43.5%           |
| PCA variance (7 components) | 70%+            |
| t-SNE KL divergence         | 1.14            |
| Overall arrest rate         | 25.2%           |
| Peak crime hour             | 12:00 & 00:00   |
| Highest crime season        | Summer          |

---

## 🛠️ Tech Stack

| Layer       | Tools                                    |
|-------------|------------------------------------------|
| Language    | Python 3.11                              |
| ML          | scikit-learn (K-Means, DBSCAN, PCA, t-SNE) |
| Viz DR      | umap-learn                               |
| Tracking    | MLflow                                   |
| App         | Streamlit                                |
| Charts      | Plotly, Matplotlib                       |
| Container   | Docker, docker-compose                   |
| Deployment  | Streamlit Cloud                          |
