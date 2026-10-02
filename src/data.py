"""Loading and cleaning the raw customer data."""
import numpy as np
import pandas as pd

from src.config import ID_COL, RAW_DATA, TARGET
from src.features import RAW_FEATURES


def load_raw(path=RAW_DATA) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types and fill gaps. Safe to call on training data or new customers."""
    df = df.copy()
    total = df["TotalCharges"].astype(str).str.strip().replace("", np.nan)
    df["TotalCharges"] = pd.to_numeric(total, errors="coerce")
    # Brand-new customers (tenure 0) have a blank TotalCharges.
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    if TARGET in df.columns and not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = (df[TARGET].astype(str).str.strip() == "Yes").astype(int)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    return X, df[TARGET]
