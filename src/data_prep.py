from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_object_dtype,
    is_string_dtype,
    is_categorical_dtype,
    is_bool_dtype,
    is_numeric_dtype,
)

TARGET_COL = "Churn"
ID_COL = "customerID"

INTERNET_ADDON_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

PHONE_RELATED_COLS = [
    "MultipleLines",
]

BASE_NUM_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def load_raw(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans + engineers features.
    Safe for:
    - full training dataframe (has Churn)
    - single-row inference dataframe (no Churn)
    """
    df = df.copy()

    # --- numeric fixes ---
    for c in BASE_NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    if "tenure" in df.columns:
        df["tenure"] = df["tenure"].fillna(0).astype(float)

    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = df["MonthlyCharges"].fillna(0.0)

    # --- normalize special category values ---
    for c in INTERNET_ADDON_COLS:
        if c in df.columns:
            df[c] = df[c].replace({"No internet service": "No"}).astype(str)

    for c in PHONE_RELATED_COLS:
        if c in df.columns:
            df[c] = df[c].replace({"No phone service": "No"}).astype(str)

    # --- target mapping (if present) ---
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)

    # =========================
    # Feature engineering
    # =========================
    # HasInternet
    if "InternetService" in df.columns:
        df["HasInternet"] = (df["InternetService"].astype(str) != "No").astype(int)

    # Contract months (numeric signal)
    if "Contract" in df.columns:
        contract_map = {"Month-to-month": 1, "One year": 12, "Two year": 24}
        df["ContractMonths"] = df["Contract"].map(contract_map).fillna(0).astype(int)

    # Tenure bands (categorical signal)
    if "tenure" in df.columns:
        bins = [-1, 6, 12, 24, 36, 48, 60, 72, 10_000]
        labels = ["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72", "73+"]
        df["TenureBand"] = pd.cut(df["tenure"], bins=bins, labels=labels).astype(str)

    # Avg monthly charges based on lifetime spend
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        denom = df["tenure"].replace(0, 1)
        df["AvgMonthlyCharges"] = (df["TotalCharges"] / denom).astype(float)
        df["ChargeGap"] = (df["MonthlyCharges"] - df["AvgMonthlyCharges"]).astype(float)
        df["ChargeRatio"] = (df["MonthlyCharges"] / (df["AvgMonthlyCharges"] + 1e-6)).astype(float)

    # Count number of internet add-on services = Yes
    present_addons = [c for c in INTERNET_ADDON_COLS if c in df.columns]
    if present_addons:
        yes_matrix = pd.DataFrame({c: (df[c].astype(str) == "Yes").astype(int) for c in present_addons})
        df["InternetAddOnCount"] = yes_matrix.sum(axis=1).astype(int)
    else:
        df["InternetAddOnCount"] = 0

    # Family flag
    if "Partner" in df.columns and "Dependents" in df.columns:
        df["HasFamily"] = ((df["Partner"].astype(str) == "Yes") | (df["Dependents"].astype(str) == "Yes")).astype(int)

    # Log transforms (often helps)
    if "MonthlyCharges" in df.columns:
        df["LogMonthlyCharges"] = np.log1p(df["MonthlyCharges"]).astype(float)
    if "TotalCharges" in df.columns:
        df["LogTotalCharges"] = np.log1p(df["TotalCharges"]).astype(float)

    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in [TARGET_COL, ID_COL]]


def get_cat_num_cols(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    cat_cols: list[str] = []
    num_cols: list[str] = []

    for c in feature_cols:
        s = df[c]
        if is_object_dtype(s) or is_string_dtype(s) or is_categorical_dtype(s) or is_bool_dtype(s):
            cat_cols.append(c)
        elif is_numeric_dtype(s):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    return cat_cols, num_cols