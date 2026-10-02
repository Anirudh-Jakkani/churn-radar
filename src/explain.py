"""SHAP explanations, aggregated back to the original (human-readable) features."""
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from src.features import original_feature


def make_explainer(model, background: np.ndarray):
    if isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, background)
    return shap.TreeExplainer(model)


def shap_values(explainer, Xt: np.ndarray) -> np.ndarray:
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):  # older per-class output
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (rows, features, classes)
        sv = sv[:, :, 1]
    return sv


def aggregate(sv: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Sum one-hot columns so each original feature gets a single contribution."""
    df = pd.DataFrame(sv, columns=feature_names)
    groups = [original_feature(n) for n in feature_names]
    return df.T.groupby(groups, sort=False).sum().T
