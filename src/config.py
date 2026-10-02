"""Central paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "telco_churn.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
REPORT_PATH = MODEL_DIR / "report.json"

TARGET = "Churn"
ID_COL = "customerID"
RANDOM_STATE = 42

# Probability bands used for the human-friendly risk label.
RISK_BANDS = [(0.6, "High"), (0.3, "Medium"), (0.0, "Low")]
