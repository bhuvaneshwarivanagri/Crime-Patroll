"""
=============================================================
PatrolIQ - Step 2: Exploratory Data Analysis (EDA)
=============================================================
Goal:
  - Analyze crime distribution across 33 crime types
  - Study geographic patterns (lat/lon)
  - Investigate temporal trends (hourly, daily, monthly, seasonal)
  - Examine arrest rates and domestic incident correlations
  - Save charts and print statistical summaries
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# ─── CONFIG ───────────────────────────────────────────────
CLEAN_PATH  = "data/cleaned_crimes.csv"
CHARTS_DIR  = "data/eda_charts"
os.makedirs(CHARTS_DIR, exist_ok=True)
# ──────────────────────────────────────────────────────────


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"Loaded {len(df):,} cleaned rows for EDA\n")
    return df


# ── 2.1  Crime Type Distribution ──────────────────────────
def plot_crime_distribution(df: pd.DataFrame) -> None:
    counts = df["Primary Type"].value_counts()
    print("── Top 10 Crime Types ──────────────────────────")
    print(counts.head(10).to_string())

    fig, ax = plt.subplots(figsize=(12, 7))
    counts.head(20).sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top 20 Crime Types in Chicago", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Incidents")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    path = f"{CHARTS_DIR}/01_crime_distribution.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\n  Chart saved → {path}")


# ── 2.2  Temporal Trends ──────────────────────────────────
def plot_temporal_trends(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Temporal Crime Patterns", fontsize=15, fontweight="bold")

    # Hourly
    hourly = df.groupby("Hour").size()
    axes[0, 0].bar(hourly.index, hourly.values, color="tomato")
    axes[0, 0].set_title("Crimes by Hour of Day")
    axes[0, 0].set_xlabel("Hour (0–23)")
    axes[0, 0].set_ylabel("Count")

    # Day of week — enforce Mon-Sun order
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
    daily = df["Day_of_Week"].value_counts().reindex(dow_order)
    axes[0, 1].bar(daily.index, daily.values, color="seagreen")
    axes[0, 1].set_title("Crimes by Day of Week")
    axes[0, 1].tick_params(axis="x", rotation=30)

    # Monthly
    monthly = df.groupby("Month").size()
    axes[1, 0].plot(monthly.index, monthly.values, marker="o", color="purple")
    axes[1, 0].set_title("Crimes by Month")
    axes[1, 0].set_xlabel("Month")
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].set_xticklabels(
        ["Jan","Feb","Mar","Apr","May","Jun",
         "Jul","Aug","Sep","Oct","Nov","Dec"], rotation=30)

    # Seasonal
    seasonal = df["Season"].value_counts()
    axes[1, 1].pie(seasonal.values, labels=seasonal.index, autopct="%1.1f%%",
                   colors=["#5b9bd5","#ed7d31","#a9d18e","#ffc000"],
                   startangle=90)
    axes[1, 1].set_title("Crimes by Season")

    plt.tight_layout()
    path = f"{CHARTS_DIR}/02_temporal_trends.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Chart saved → {path}")

    # Print summaries
    print("\n── Hourly Top 5 ────────────────────────────────")
    print(hourly.sort_values(ascending=False).head(5).to_string())
    print("\n── Seasonal Breakdown ──────────────────────────")
    print(seasonal.to_string())


# ── 2.3  Arrest Rate & Domestic Incidents ─────────────────
def plot_arrest_domestic(df: pd.DataFrame) -> None:
    arrest_rate   = df["Arrest"].mean() * 100
    domestic_rate = df["Domestic"].mean() * 100
    print(f"\n── Arrest & Domestic Stats ─────────────────────")
    print(f"  Overall arrest rate   : {arrest_rate:.2f}%")
    print(f"  Domestic incident rate: {domestic_rate:.2f}%")

    # Arrest rate per crime type
    arrest_by_type = (df.groupby("Primary Type")["Arrest"]
                        .mean()
                        .sort_values(ascending=False)
                        .head(15) * 100)

    fig, ax = plt.subplots(figsize=(12, 7))
    arrest_by_type.sort_values().plot(kind="barh", ax=ax, color="coral")
    ax.set_title("Arrest Rate by Crime Type (Top 15)", fontsize=13,
                 fontweight="bold")
    ax.set_xlabel("Arrest Rate (%)")
    plt.tight_layout()
    path = f"{CHARTS_DIR}/03_arrest_rates.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Chart saved → {path}")


# ── 2.4  Geographic Scatter ───────────────────────────────
def plot_geographic(df: pd.DataFrame) -> None:
    # Sample 50,000 points so the scatter is readable
    sample = df.sample(50_000, random_state=42)

    fig, ax = plt.subplots(figsize=(9, 11))
    sc = ax.scatter(sample["Longitude"], sample["Latitude"],
                    c=sample["Crime_Severity_Score"],
                    cmap="RdYlGn_r", alpha=0.25, s=1.5)
    plt.colorbar(sc, ax=ax, label="Crime Severity Score")
    ax.set_title("Geographic Distribution of Crimes\n(50K sample, colored by severity)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    path = f"{CHARTS_DIR}/04_geographic_scatter.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Chart saved → {path}")


# ── 2.5  Year-over-Year Trend ─────────────────────────────
def plot_yearly_trend(df: pd.DataFrame) -> None:
    yearly = df.groupby("Year").size()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(yearly.index, yearly.values, color="slateblue")
    ax.set_title("Crime Count by Year", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Crimes")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    path = f"{CHARTS_DIR}/05_yearly_trend.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Chart saved → {path}")

    print("\n── Year-over-Year Summary ──────────────────────")
    print(yearly.to_string())


# ── 2.6  Statistical Summary ──────────────────────────────
def print_statistics(df: pd.DataFrame) -> None:
    print("\n── Overall Statistical Summary ─────────────────")
    cols = ["Hour", "Month", "Crime_Severity_Score",
            "Latitude", "Longitude", "District"]
    print(df[cols].describe().round(3).to_string())
    print(f"\n  Unique crime types    : {df['Primary Type'].nunique()}")
    print(f"  Unique districts      : {df['District'].nunique()}")
    print(f"  Unique community areas: {df['Community Area'].nunique()}")
    print(f"  Date range            : {df['Date'].min().date()} → "
          f"{df['Date'].max().date()}")


# ─── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    df = load(CLEAN_PATH)

    print("=" * 52)
    print("  EDA Section 1 — Crime Type Distribution")
    print("=" * 52)
    plot_crime_distribution(df)

    print("\n" + "=" * 52)
    print("  EDA Section 2 — Temporal Trends")
    print("=" * 52)
    plot_temporal_trends(df)

    print("\n" + "=" * 52)
    print("  EDA Section 3 — Arrest & Domestic Rates")
    print("=" * 52)
    plot_arrest_domestic(df)

    print("\n" + "=" * 52)
    print("  EDA Section 4 — Geographic Scatter")
    print("=" * 52)
    plot_geographic(df)

    print("\n" + "=" * 52)
    print("  EDA Section 5 — Year-over-Year Trend")
    print("=" * 52)
    plot_yearly_trend(df)

    print_statistics(df)
    print("\n✅ Step 2 Complete — EDA charts saved to", CHARTS_DIR)
