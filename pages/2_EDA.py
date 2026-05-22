"""Page 2 — Exploratory Data Analysis"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.title("📊 Exploratory Data Analysis")

@st.cache_data
def load_meta():
    return pd.read_csv("data/metadata.csv", parse_dates=["Date"])

meta = load_meta()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Crime Types", "Temporal Trends", "Arrest & Domestic", "Geographic"])

# ── Tab 1: Crime distribution ─────────────────────────────
with tab1:
    st.subheader("Crime Type Distribution")
    n = st.slider("Show top N crime types", 5, 32, 15)
    counts = meta["Primary Type"].value_counts().head(n).reset_index()
    counts.columns = ["Crime Type", "Count"]
    fig = px.bar(counts.sort_values("Count"), x="Count", y="Crime Type",
                 orientation="h", color="Count",
                 color_continuous_scale="Blues",
                 title=f"Top {n} Crime Types")
    fig.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Crime Severity Distribution**")
    sev_counts = meta["Crime_Severity_Score"].value_counts().sort_index()
    fig2 = px.bar(x=sev_counts.index, y=sev_counts.values,
                  labels={"x":"Severity Score","y":"Count"},
                  color=sev_counts.values,
                  color_continuous_scale="RdYlGn_r",
                  title="Crime Count by Severity Score (1=Minor, 10=Severe)")
    st.plotly_chart(fig2, use_container_width=True)

# ── Tab 2: Temporal ───────────────────────────────────────
with tab2:
    st.subheader("Temporal Crime Patterns")
    col1, col2 = st.columns(2)

    with col1:
        hourly = meta["Date"].dt.hour.value_counts().sort_index().reset_index()
        hourly.columns = ["Hour", "Count"]
        fig = px.bar(hourly, x="Hour", y="Count",
                     title="Crimes by Hour of Day",
                     color="Count", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        dow_order = ["Monday","Tuesday","Wednesday","Thursday",
                     "Friday","Saturday","Sunday"]
        daily = meta["Date"].dt.day_name().value_counts().reindex(
            dow_order).reset_index()
        daily.columns = ["Day", "Count"]
        fig = px.bar(daily, x="Day", y="Count",
                     title="Crimes by Day of Week",
                     color="Count", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    monthly = meta["Date"].dt.month.value_counts().sort_index().reset_index()
    monthly.columns = ["Month", "Count"]
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly["Month"] = monthly["Month"].map(month_names)
    fig = px.line(monthly, x="Month", y="Count", markers=True,
                  title="Crimes by Month (Seasonal Trend)")
    st.plotly_chart(fig, use_container_width=True)

    # Season pie
    season_counts = meta["Season"].value_counts().reset_index()
    season_counts.columns = ["Season", "Count"]
    fig = px.pie(season_counts, names="Season", values="Count",
                 title="Crime Distribution by Season",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: Arrest & Domestic ──────────────────────────────
with tab3:
    st.subheader("Arrest Rate & Domestic Incidents")

    col1, col2 = st.columns(2)
    col1.metric("Overall Arrest Rate",    f"{meta['Arrest'].mean()*100:.1f}%")
    col2.metric("Domestic Incident Rate", f"{meta['Domestic'].mean()*100:.1f}%")

    arr_rate = (meta.groupby("Primary Type")["Arrest"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(15)
                    .reset_index())
    arr_rate.columns = ["Crime Type", "Arrest Rate"]
    arr_rate["Arrest Rate"] = (arr_rate["Arrest Rate"] * 100).round(2)
    fig = px.bar(arr_rate.sort_values("Arrest Rate"),
                 x="Arrest Rate", y="Crime Type",
                 orientation="h",
                 title="Arrest Rate by Crime Type (Top 15, %)",
                 color="Arrest Rate",
                 color_continuous_scale="RdYlGn")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 4: Geographic ─────────────────────────────────────
with tab4:
    st.subheader("Geographic Crime Distribution")
    st.info("Showing a 10,000-record sample for performance.")
    sample = meta.dropna(subset=["Latitude","Longitude"]).sample(
        10_000, random_state=42)

    crime_filter = st.multiselect(
        "Filter by crime type (empty = all)",
        options=sorted(meta["Primary Type"].unique()),
        default=[])
    if crime_filter:
        sample = sample[sample["Primary Type"].isin(crime_filter)]

    fig = px.scatter_mapbox(
        sample,
        lat="Latitude", lon="Longitude",
        color="Crime_Severity_Score",
        color_continuous_scale="RdYlGn_r",
        hover_data=["Primary Type","Season","Arrest"],
        zoom=10, height=600,
        title="Crime Map — Colored by Severity",
        mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)
