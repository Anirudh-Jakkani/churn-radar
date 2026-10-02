"""Train, compare, tune and save the churn model.

Usage:  python -m src.train [--trials 40]
"""
import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import optuna
import shap
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, cross_val_score, train_test_split,
)
from xgboost import XGBClassifier

from src.config import MODEL_DIR, MODEL_PATH, RANDOM_STATE, REPORT_PATH
from src.data import clean, load_raw, split_xy
from src.explain import aggregate, make_explainer, shap_values
from src.features import build_pipeline


def candidate_models(pos_weight: float) -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15, class_weight="balanced",
            random_state=RANDOM_STATE, verbose=-1),
    }


def tune_xgboost(X, y, cv, pos_weight: float, n_trials: int):
    def objective(trial):
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            scale_pos_weight=pos_weight, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        return cross_val_score(build_pipeline(model), X, y, cv=cv, scoring="roc_auc").mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    model = XGBClassifier(**study.best_params, scale_pos_weight=pos_weight,
                          eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
    return model, study.best_value, study.best_params


def best_f1_threshold(y_true, proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[np.nanargmax(f1[:-1])])


def main(n_trials: int) -> None:
    df = clean(load_raw())
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())
    print(f"Train {len(X_tr)} / Test {len(X_te)} | churn rate {y.mean():.1%}")

    models = candidate_models(pos_weight)
    leaderboard = {}
    for name, model in models.items():
        scores = cross_val_score(build_pipeline(model), X_tr, y_tr, cv=cv, scoring="roc_auc")
        leaderboard[name] = {"cv_roc_auc": round(scores.mean(), 4), "cv_std": round(scores.std(), 4)}
        print(f"  {name:<22} CV ROC-AUC {scores.mean():.4f} ± {scores.std():.4f}")

    if n_trials > 0:
        print(f"Tuning XGBoost with Optuna ({n_trials} trials)...")
        tuned, score, params = tune_xgboost(X_tr, y_tr, cv, pos_weight, n_trials)
        models["XGBoost (tuned)"] = tuned
        leaderboard["XGBoost (tuned)"] = {"cv_roc_auc": round(score, 4), "cv_std": None, "params": params}
        print(f"  {'XGBoost (tuned)':<22} CV ROC-AUC {score:.4f}")

    best_name = max(leaderboard, key=lambda k: leaderboard[k]["cv_roc_auc"])
    print(f"Best model: {best_name}")
    pipeline = build_pipeline(clone(models[best_name]))

    # Pick the decision threshold on out-of-fold predictions (never on the test set).
    oof = cross_val_predict(pipeline, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    threshold = best_f1_threshold(y_tr, oof)

    pipeline.fit(X_tr, y_tr)
    proba = pipeline.predict_proba(X_te)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    metrics = {
        "roc_auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
    }
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    print("Test metrics @ threshold %.3f: %s" % (threshold, metrics))

    fpr, tpr, _ = roc_curve(y_te, proba)
    idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 120)).astype(int)

    # Global explainability on the test set.
    prep = pipeline[:-1]
    feature_names = pipeline.named_steps["prep"].get_feature_names_out().tolist()
    background = shap.sample(prep.transform(X_tr), 200, random_state=RANDOM_STATE)
    explainer = make_explainer(pipeline.named_steps["model"], background)
    contrib = aggregate(shap_values(explainer, prep.transform(X_te)), feature_names)
    importance = contrib.abs().mean().sort_values(ascending=False)

    report = {
        "model_name": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": round(threshold, 4),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "churn_rate": round(float(y.mean()), 4),
        "metrics": metrics,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()},
        "leaderboard": leaderboard,
        "feature_importance": importance.round(4).to_dict(),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "threshold": threshold,
        "feature_names": feature_names,
        "background": background,
        "report": report,
    }, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40, help="Optuna trials (0 to skip tuning)")
    main(parser.parse_args().trials)
