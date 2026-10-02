"""Feature engineering and the preprocessing pipeline."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Allowed values for every categorical input (shared by the API schema and the UI).
CATEGORY_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}
INTERNET_ADDONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
SERVICE_COLS = ["PhoneService", "MultipleLines", *INTERNET_ADDONS]

RAW_NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
RAW_CATEGORICAL = list(CATEGORY_OPTIONS)
RAW_FEATURES = RAW_NUMERIC + RAW_CATEGORICAL

ENGINEERED_NUMERIC = ["avg_monthly_spend", "charge_ratio", "num_services"]
ENGINEERED_CATEGORICAL = ["tenure_bucket", "auto_pay"]

NUMERIC = RAW_NUMERIC + ENGINEERED_NUMERIC
CATEGORICAL = RAW_CATEGORICAL + ENGINEERED_CATEGORICAL

TENURE_BINS = [-1, 6, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-6m", "6-12m", "1-2y", "2-4y", "4y+"]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["avg_monthly_spend"] = X["TotalCharges"] / X["tenure"].clip(lower=1)
    # >1 means the current bill is higher than the customer's historical average.
    X["charge_ratio"] = (X["MonthlyCharges"] / X["avg_monthly_spend"].replace(0, np.nan)).fillna(1.0)
    X["num_services"] = (X[SERVICE_COLS] == "Yes").sum(axis=1)
    X["tenure_bucket"] = pd.cut(X["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    X["auto_pay"] = np.where(X["PaymentMethod"].str.contains("automatic"), "Yes", "No")
    return X


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


def build_pipeline(model) -> Pipeline:
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", build_preprocessor()),
        ("model", model),
    ])


def original_feature(transformed_name: str) -> str:
    """Map an encoded column name (e.g. 'cat__Contract_Two year') back to 'Contract'."""
    kind, _, name = transformed_name.partition("__")
    if kind == "num":
        return name
    matches = [c for c in CATEGORICAL if name.startswith(c + "_")]
    return max(matches, key=len) if matches else name
