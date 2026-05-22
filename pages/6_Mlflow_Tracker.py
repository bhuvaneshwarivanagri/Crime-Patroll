"""Page 6 — MLflow Experiment Tracker"""
import streamlit as st
import pandas as pd
import plotly.express as px
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Crime_Patrol")

st.title("📈 MLflow Experiment Tracker")

st.markdown("""
All clustering and dimensionality reduction experiments are logged to **MLflow**.
This page summarizes tracked runs, parameters, and metrics.
""")

@st.cache_data
def fetch_runs():
    client = mlflow.tracking.MlflowClient()
    rows = []
    for exp in client.search_experiments():
        for run in client.search_runs(experiment_ids=[exp.experiment_id]):
            row = {
                "Experiment" : exp.name,
                "Run Name"   : run.info.run_name,
                "Status"     : run.info.status,
            }
            row.update(run.data.params)
            row.update({f"metric_{k}": round(v, 4)
                        for k, v in run.data.metrics.items()})
            rows.append(row)
    return pd.DataFrame(rows)

try:
    df_runs = fetch_runs()

    st.subheader("All Logged Experiments")
    experiments = ["All"] + sorted(df_runs["Experiment"].unique().tolist())
    chosen_exp  = st.selectbox("Filter by experiment", experiments)
    if chosen_exp != "All":
        df_runs = df_runs[df_runs["Experiment"] == chosen_exp]

    st.dataframe(df_runs, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(df_runs)} runs")

    # Metric visualisation
    metric_cols = [c for c in df_runs.columns if c.startswith("metric_")]
    if metric_cols and "Run Name" in df_runs.columns:
        st.subheader("Metric Comparison")
        metric = st.selectbox("Select metric to compare", metric_cols)
        df_plot = df_runs.dropna(subset=[metric])
        if not df_plot.empty:
            fig = px.bar(df_plot, x="Run Name", y=metric,
                         color="Experiment",
                         title=f"{metric} across runs",
                         barmode="group")
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Could not connect to MLflow: {e}")
    st.info("Make sure you have run Steps 4 and 5 to populate the MLflow store.")

st.markdown("---")
st.subheader("📋 Manual Experiment Summary")

summary = {
    "Experiment"         : ["Geographic","Geographic","Geographic",
                            "Temporal","Temporal","Temporal","Temporal",
                            "Dim Reduction","Dim Reduction","Dim Reduction"],
    "Algorithm"          : ["K-Means","DBSCAN","Hierarchical",
                            "K-Means k=3","K-Means k=4","K-Means k=5","K-Means k=6",
                            "PCA","t-SNE","UMAP"],
    "Key Metric"         : ["Silhouette","Silhouette","Silhouette",
                            "Silhouette","Silhouette","Silhouette","Silhouette",
                            "Variance%","KL Divergence","N/A"],
    "Value"              : [0.487, -1.0, 0.460,
                            0.328, 0.325, 0.312, 0.340,
                            43.53, 1.14, None],
    "Selected?"          : ["✅ Best","❌","2nd",
                            "","","","✅ Best",
                            "✅","✅","✅"],
}
st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

st.markdown("""
**How to launch the MLflow UI:**
```bash
In bash enter
mlflow ui 
```

""")
