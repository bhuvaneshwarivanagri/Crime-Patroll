"""
=============================================================
PatrolIQ - Step 1: Data Acquisition & Preprocessing
=============================================================
Goal:
  - Load 500,000 crime records from patrol.csv
  - Clean missing values and inconsistencies
  - Extract temporal features (hour, day_of_week, month, season)
  - Apply data quality assessment and validation
  - Save cleaned data for use in later steps
=============================================================
"""

import pandas as pd
import numpy as np
import os

# ─── CONFIG ───────────────────────────────────────────────
RAW_PATH   = "data/patrol.csv"         # input
CLEAN_PATH = "data/cleaned_crimes.csv" # output
# ──────────────────────────────────────────────────────────


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV file."""
    print(f"[1/6] Loading data from: {path}")
    df = pd.read_csv(path, index_col=0)
    print(f"      Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def assess_quality(df: pd.DataFrame) -> None:
    """Print a data quality report."""
    print("\n[2/6] Data Quality Assessment")
    print("-" * 45)
    total = len(df)
    null_summary = df.isnull().sum()
    null_pct     = (null_summary / total * 100).round(2)
    quality_df   = pd.DataFrame({"Missing Count": null_summary,
                                 "Missing %": null_pct})
    quality_df   = quality_df[quality_df["Missing Count"] > 0]
    print(quality_df.to_string())
    print(f"\n  Total rows  : {total:,}")
    print(f"  Total cols  : {df.shape[1]}")
    print(f"  Duplicate rows: {df.duplicated().sum():,}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning steps:
      1. Drop rows where Latitude/Longitude are missing
         (coordinates are essential for geographic clustering)
      2. Fill remaining nulls in District, Ward, Community Area
         with their median values
      3. Fill missing Location Description with 'UNKNOWN'
      4. Remove exact duplicate rows
    """
    print("\n[3/6] Cleaning Data")
    original_len = len(df)

    # 3a. Drop rows without geographic coordinates
    df = df.dropna(subset=["Latitude", "Longitude"])
    print(f"      After dropping missing coords : {len(df):,} rows")

    # 3b. Fill numeric admin columns with median
    for col in ["District", "Ward", "Community Area"]:
        median_val = df[col].median()
        df[col]    = df[col].fillna(median_val)

    # 3c. Fill categorical nulls
    df["Location Description"] = df["Location Description"].fillna("UNKNOWN")

    # 3d. Remove duplicates
    df = df.drop_duplicates()
    print(f"      After removing duplicates     : {len(df):,} rows")
    print(f"      Rows removed in total         : {original_len - len(df):,}")
    return df


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the Date column and derive:
      Hour        - 0 to 23
      Day_of_Week - Monday … Sunday
      Month       - 1 to 12
      Season      - Winter / Spring / Summer / Fall
      Is_Weekend  - True if Saturday or Sunday
    """
    print("\n[4/6] Extracting Temporal Features")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p",
                                errors="coerce")

    df["Hour"]        = df["Date"].dt.hour
    df["Day_of_Week"] = df["Date"].dt.day_name()
    df["Month"]       = df["Date"].dt.month
    df["Is_Weekend"]  = df["Date"].dt.dayofweek >= 5   # 5=Sat, 6=Sun

    # Season mapping by month
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
         3: "Spring", 4: "Spring", 5: "Spring",
         6: "Summer", 7: "Summer", 8: "Summer",
         9: "Fall",  10: "Fall",  11: "Fall"
    }
    df["Season"] = df["Month"].map(season_map)

    print(f"      Hour range   : {df['Hour'].min()} – {df['Hour'].max()}")
    print(f"      Month range  : {df['Month'].min()} – {df['Month'].max()}")
    print(f"      Seasons      : {df['Season'].unique().tolist()}")
    print(f"      Weekend rows : {df['Is_Weekend'].sum():,}")
    return df


def add_crime_severity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a numeric severity score to each crime type.
    Scale: 1 (minor) → 10 (most severe)
    """
    print("\n[5/6] Adding Crime Severity Score")

    severity_map = {
        "HOMICIDE"                   : 10,
        "CRIM SEXUAL ASSAULT"        : 9,
        "KIDNAPPING"                 : 9,
        "ARSON"                      : 8,
        "ROBBERY"                    : 8,
        "ASSAULT"                    : 7,
        "BATTERY"                    : 7,
        "WEAPONS VIOLATION"          : 7,
        "BURGLARY"                   : 6,
        "MOTOR VEHICLE THEFT"        : 6,
        "STALKING"                   : 6,
        "INTIMIDATION"               : 5,
        "THEFT"                      : 5,
        "CRIMINAL DAMAGE"            : 4,
        "SEX OFFENSE"                : 4,
        "DECEPTIVE PRACTICE"         : 4,
        "NARCOTICS"                  : 4,
        "OFFENSE INVOLVING CHILDREN" : 4,
        "CRIMINAL TRESPASS"          : 3,
        "PUBLIC PEACE VIOLATION"     : 3,
        "INTERFERENCE WITH PUBLIC OFFICER": 3,
        "LIQUOR LAW VIOLATION"       : 2,
        "GAMBLING"                   : 2,
        "OBSCENITY"                  : 2,
        "OTHER OFFENSE"              : 2,
    }
    # Default severity for unmapped types
    df["Crime_Severity_Score"] = df["Primary Type"].map(severity_map).fillna(2)

    print(f"      Severity range : {df['Crime_Severity_Score'].min()} – "
          f"{df['Crime_Severity_Score'].max()}")
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    """Save the cleaned DataFrame to CSV."""
    print(f"\n[6/6] Saving cleaned data to: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"      Saved {len(df):,} rows × {df.shape[1]} columns ✓")


# ─── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data(RAW_PATH)
    assess_quality(df)
    df = clean_data(df)
    df = extract_temporal_features(df)
    df = add_crime_severity(df)

    print("\n── Final Column List ─────────────────────────────")
    print(df.dtypes.to_string())

    print("\n── Sample Rows ───────────────────────────────────")
    print(df[["Primary Type", "Hour", "Day_of_Week", "Season",
              "Is_Weekend", "Crime_Severity_Score",
              "Latitude", "Longitude"]].head(5).to_string())

    save_data(df, CLEAN_PATH)
    print("\n✅ Step 1 Complete — Data ready for EDA & Feature Engineering")
