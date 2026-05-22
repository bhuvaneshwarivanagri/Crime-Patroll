"""Page 4 — Temporal Clustering"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.title("⏰ Temporal Crime Pattern Clustering")

@st.cache_data
def load():
    meta = pd.read_csv("data/metadata.csv", parse_dates=["Date"])
    temp = pd.read_csv("data/temporal_cluster_results.csv",
                       parse_dates=["Date"])
    return meta, temp

meta, temp = load()

st.markdown("""
Temporal clustering groups crimes by **when** they occur —
discovering patterns like *late-night violence*, *rush-hour theft*,
and *weekend domestic incidents*.
""")

tab1, tab2, tab3 = st.tabs(
    ["Cluster Overview", "Hourly Heatmap", "Seasonal & Weekly"])

# ── Tab 1: Overview ───────────────────────────────────────
with tab1:
    st.subheader("Temporal Cluster Chart")
    st.image("data/cluster_charts/temporal_kmeans.png",
             caption="Crime Count by Hour — colored by Temporal Cluster",
             use_container_width=True)

    if "Temporal_Cluster" in temp.columns and "Hour_val" in temp.columns:
        st.subheader("Cluster Profiles")
        profile = temp.groupby("Temporal_Cluster").agg(
            Size       =("Primary Type","count"),
            Avg_Hour   =("Hour_val",   "mean"),
            Top_Crime  =("Primary Type", lambda x: x.value_counts().idxmax()),
            Arrest_Rate=("Arrest",    lambda x: x.mean()*100),
        ).round(2).reset_index()
        profile.columns = ["Cluster","Size","Avg Hour",
                           "Dominant Crime","Arrest Rate (%)"]

        def label_cluster(row):
            h = row["Avg Hour"]
            if h < 6:   return "🌙 Late Night"
            if h < 12:  return "🌅 Morning"
            if h < 17:  return "☀️ Afternoon"
            return "🌆 Evening"
        profile["Time Period"] = profile.apply(label_cluster, axis=1)

        st.dataframe(profile, use_container_width=True, hide_index=True)

        fig = px.bar(profile, x="Cluster", y="Size",
                     color="Avg Hour", color_continuous_scale="twilight",
                     title="Temporal Cluster Sizes (colored by avg hour)",
                     text="Time Period")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Hourly Heatmap ─────────────────────────────────
with tab2:
    st.subheader("Interactive Hourly Crime Heatmap")
    crime_types = ["All"] + sorted(meta["Primary Type"].unique().tolist())
    chosen = st.selectbox("Filter by crime type", crime_types)

    df_h = meta.copy()
    df_h["Hour"] = df_h["Date"].dt.hour
    df_h["DOW"]  = df_h["Date"].dt.day_name()
    if chosen != "All":
        df_h = df_h[df_h["Primary Type"] == chosen]

    dow_order = ["Monday","Tuesday","Wednesday","Thursday",
                 "Friday","Saturday","Sunday"]
    heatmap = (df_h.groupby(["DOW","Hour"])
                   .size()
                   .reset_index(name="Count"))
    heatmap["DOW"] = pd.Categorical(heatmap["DOW"],
                                    categories=dow_order, ordered=True)
    heatmap = heatmap.sort_values("DOW")

    fig = px.density_heatmap(
        heatmap, x="Hour", y="DOW", z="Count",
        color_continuous_scale="YlOrRd",
        title=f"Crime Frequency Heatmap — {chosen}",
        labels={"Hour":"Hour of Day","DOW":"Day of Week","Count":"Crimes"})
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Peak hour box
    peak_hour = df_h["Hour"].value_counts().idxmax()
    peak_count = df_h["Hour"].value_counts().max()
    st.info(f"🔴 **Peak hour:** {peak_hour}:00  |  "
            f"**Crimes at this hour:** {peak_count:,}")

# ── Tab 3: Seasonal & Weekly ──────────────────────────────
with tab3:
    st.subheader("Seasonal & Weekly Crime Trends")
    col1, col2 = st.columns(2)

    with col1:
        season = meta["Season"].value_counts().reset_index()
        season.columns = ["Season","Count"]
        fig = px.pie(season, names="Season", values="Count",
                     title="Crime by Season",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        dow_order = ["Monday","Tuesday","Wednesday","Thursday",
                     "Friday","Saturday","Sunday"]
        wday = meta["Date"].dt.day_name().value_counts().reindex(
            dow_order).reset_index()
        wday.columns = ["Day","Count"]
        fig = px.bar(wday, x="Day", y="Count",
                     color="Count", color_continuous_scale="Purples",
                     title="Crime by Day of Week")
        st.plotly_chart(fig, use_container_width=True)

    # Year trend
    yearly = meta["Date"].dt.year.value_counts().sort_index().reset_index()
    yearly.columns = ["Year","Count"]
    fig = px.area(yearly, x="Year", y="Count",
                  title="Crime Trend by Year (2001–2026)",
                  color_discrete_sequence=["#1f77b4"])
    st.plotly_chart(fig, use_container_width=True)
