"""Load the trained model and score / explain customers."""
import joblib
import numpy as np
import pandas as pd

from src.config import ID_COL, MODEL_PATH, RISK_BANDS, TARGET
from src.data import clean, validate_columns
from src.explain import aggregate, make_explainer, shap_values
from src.features import add_features
from src.retention import recommend


def risk_level(p: float) -> str:
    return next(label for cutoff, label in RISK_BANDS if p >= cutoff)


def _plain(v):
    if isinstance(v, np.generic):
        v = v.item()
    return round(v, 2) if isinstance(v, float) else v


class ChurnPredictor:
    def __init__(self, model_path=MODEL_PATH):
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.threshold = bundle["threshold"]
        self.feature_names = bundle["feature_names"]
        self.background = bundle["background"]
        self.report = bundle["report"]
        self._explainer = None

    @property
    def explainer(self):
        if self._explainer is None:
            self._explainer = make_explainer(self.pipeline.named_steps["model"], self.background)
        return self._explainer

    @staticmethod
    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        validate_columns(df)
        return clean(df).drop(columns=[TARGET, ID_COL], errors="ignore")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        proba = self.pipeline.predict_proba(self.prepare(df))[:, 1]
        return pd.DataFrame({
            "churn_probability": proba.round(4),
            "will_churn": proba >= self.threshold,
            "risk_level": [risk_level(p) for p in proba],
        }, index=df.index)

    def explain(self, df: pd.DataFrame, top_k: int | None = None) -> list[list[dict]]:
        """Per customer: features sorted by |impact| (impact > 0 pushes toward churn)."""
        X = self.prepare(df)
        contrib = aggregate(shap_values(self.explainer, self.pipeline[:-1].transform(X)),
                            self.feature_names)
        feats = add_features(X)
        results = []
        for i in range(len(X)):
            row = contrib.iloc[i].sort_values(key=np.abs, ascending=False)
            if top_k:
                row = row.head(top_k)
            results.append([
                {"feature": f, "value": _plain(feats.iloc[i][f]), "impact": round(float(v), 4)}
                for f, v in row.items()
            ])
        return results

    def analyze(self, customer: dict) -> dict:
        """Everything the UI/API needs for one customer."""
        df = pd.DataFrame([customer])
        pred = self.predict(df).iloc[0]
        reasons = self.explain(df)[0]
        features = add_features(self.prepare(df)).iloc[0].to_dict()
        return {
            "churn_probability": float(pred["churn_probability"]),
            "will_churn": bool(pred["will_churn"]),
            "risk_level": pred["risk_level"],
            "threshold": round(self.threshold, 4),
            "reasons": reasons,
            "recommended_actions": recommend(features, reasons),
        }
