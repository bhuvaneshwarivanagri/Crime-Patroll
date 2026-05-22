"""
=============================================================
PatrolIQ - Step 3: Feature Engineering
=============================================================
Goal:
  - Encode categorical features (crime type, location desc, season)
  - Normalize geographic coordinates
  - Bin geographic coordinates into grid cells
  - Create final numeric feature matrix for ML algorithms
  - Save feature matrix + metadata for Steps 4 & 5
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os, pickle

# ─── CONFIG ───────────────────────────────────────────────
CLEAN_PATH    = "data/cleaned_crimes.csv"
FEATURES_PATH = "data/features.csv"      # main feature matrix
META_PATH     = "data/metadata.csv"       # columns kept for display
SCALER_PATH   = "data1/scaler.pkl"
ENCODER_PATH  = "data1/encoders.pkl"
# ──────────────────────────────────────────────────────────

DOW_ORDER = ["Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Saturday", "Sunday"]
SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(r"C:\Users\bhuva\OneDrive\Desktop\new desktop\desktop\DS PORTAL\patrol coding files\data\cleaned_crimes.csv", parse_dates=["Date"])
    print(f"Loaded {len(df):,} rows for feature engineering")
    return df


# ── 3.1  Temporal numeric encoding ────────────────────────
def encode_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Day_of_Week → integer 0-6
    Season      → integer 0-3
    Is_Weekend  → 0 / 1
    """
    df = df.copy()
    df["DayOfWeek_Num"] = df["Day_of_Week"].map(
        {d: i for i, d in enumerate(DOW_ORDER)}).fillna(0).astype(int)
    df["Season_Num"] = df["Season"].map(
        {s: i for i, s in enumerate(SEASON_ORDER)}).fillna(0).astype(int)
    df["Is_Weekend_Num"] = df["Is_Weekend"].astype(int)
    return df


# ── 3.2  Label-encode high-cardinality categoricals ───────
def label_encode_categoricals(df: pd.DataFrame) -> tuple:
    """
    Encode Primary Type and Location Description via LabelEncoder.
    Returns (df, encoders_dict)
    """
    encoders = {}
    df = df.copy()
    for col in ["Primary Type", "Location Description"]:
        le = LabelEncoder()
        df[f"{col}_Enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  Encoded '{col}' → {len(le.classes_)} unique classes")
    return df, encoders


# ── 3.3  Coordinate binning ───────────────────────────────
def bin_coordinates(df: pd.DataFrame,
                    lat_bins: int = 20,
                    lon_bins: int = 20) -> pd.DataFrame:
    """
    Divide Chicago lat/lon space into an N×N grid.
    Each cell gets an integer grid_id used as a location feature.
    """
    df = df.copy()
    df["Lat_Bin"] = pd.cut(df["Latitude"],  bins=lat_bins, labels=False)
    df["Lon_Bin"] = pd.cut(df["Longitude"], bins=lon_bins, labels=False)
    df["Lat_Bin"] = df["Lat_Bin"].fillna(0).astype(int)
    df["Lon_Bin"] = df["Lon_Bin"].fillna(0).astype(int)
    df["Grid_ID"] = df["Lat_Bin"] * lon_bins + df["Lon_Bin"]
    print(f"  Created {lat_bins}×{lon_bins} geographic grid "
          f"({df['Grid_ID'].nunique()} occupied cells)")
    return df


# ── 3.4  Normalize geo coordinates ───────────────────────
def normalize_coords(df: pd.DataFrame) -> tuple:
    """MinMax-scale Lat/Lon to [0,1] for distance-based algorithms."""
    scaler = MinMaxScaler()
    df = df.copy()
    df[["Lat_Scaled", "Lon_Scaled"]] = scaler.fit_transform(
        df[["Latitude", "Longitude"]])
    print(f"  Normalized Latitude  → [{df['Lat_Scaled'].min():.4f}, "
          f"{df['Lat_Scaled'].max():.4f}]")
    print(f"  Normalized Longitude → [{df['Lon_Scaled'].min():.4f}, "
          f"{df['Lon_Scaled'].max():.4f}]")
    return df, scaler


# ── 3.5  Build final feature matrix ───────────────────────
FEATURE_COLS = [
    # Temporal
    "Hour", "DayOfWeek_Num", "Month", "Season_Num",
    "Is_Weekend_Num",
    # Crime
    "Primary Type_Enc", "Location Description_Enc",
    "Crime_Severity_Score",
    "Arrest",             # bool → 0/1 automatically
    "Domestic",
    # Geographic
    "Lat_Scaled", "Lon_Scaled",
    "Lat_Bin", "Lon_Bin", "Grid_ID",
    "District", "Beat", "Community Area", "Ward",
]

META_COLS = [
    "ID", "Date", "Primary Type", "Description",
    "Location Description", "Season", "Day_of_Week",
    "Latitude", "Longitude", "Year", "Arrest", "Domestic",
    "Crime_Severity_Score",
]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    # Booleans to int
    for col in ["Arrest", "Domestic"]:
        X[col] = X[col].astype(int)
    # Drop any remaining NaN rows
    before = len(X)
    X = X.dropna()
    dropped = before - len(X)
    if dropped:
        print(f"  Dropped {dropped:,} rows with remaining NaNs")
    print(f"  Feature matrix shape: {X.shape}")
    return X


def save_artifacts(X: pd.DataFrame,
                   meta: pd.DataFrame,
                   scaler: MinMaxScaler,
                   encoders: dict) -> None:
    X.to_csv(FEATURES_PATH, index=False)
    meta.to_csv(META_PATH,   index=False)
    with open(SCALER_PATH,  "wb") as f:
        pickle.dump(scaler, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoders, f)
    print(f"\n  Saved features  → {FEATURES_PATH}")
    print(f"  Saved metadata  → {META_PATH}")
    print(f"  Saved scaler    → {SCALER_PATH}")
    print(f"  Saved encoders  → {ENCODER_PATH}")


# ─── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    df = load(CLEAN_PATH)

    print("\n[1/5] Encoding temporal features")
    df = encode_temporal(df)

    print("\n[2/5] Label-encoding categoricals")
    df, encoders = label_encode_categoricals(df)

    print("\n[3/5] Binning coordinates into geographic grid")
    df = bin_coordinates(df)

    print("\n[4/5] Normalizing geographic coordinates")
    df, scaler = normalize_coords(df)

    print("\n[5/5] Building final feature matrix")
    X    = build_feature_matrix(df)
    meta = df[META_COLS].loc[X.index]   # keep same rows as X

    print("\n── Feature Columns ──────────────────────────────")
    for i, c in enumerate(X.columns, 1):
        print(f"  {i:2d}. {c}")

    save_artifacts(X, meta, scaler, encoders)
    print("\n✅ Step 3 Complete — Feature matrix ready for clustering")
