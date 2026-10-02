# 📡 ChurnRadar: Customer Churn Prediction

Predicts whether a customer is likely to churn, explains **why** (SHAP), and suggests **what to do** about it (a retention playbook).
Trained on the IBM Telco Customer Churn dataset (7,043 customers, 26.5% churn).

## Results (held-out test set, 1,409 customers)

| Metric | Score |
|---|---|
| ROC-AUC | 0.846 |
| Recall (churners caught) | 0.727 |
| Precision | 0.571 |
| F1 | 0.640 |
| PR-AUC | 0.666 |

Best model: **XGBoost tuned with Optuna** (CV ROC-AUC 0.850). It beat Logistic Regression (0.847), Random Forest (0.845), XGBoost with default settings (0.844) and LightGBM (0.841).
The decision threshold (0.60) maximizes F1 on cross-validated *training* predictions, so the test set is never used to tune it.

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate      # (Windows)  or: source .venv/bin/activate
pip install -r requirements.txt

python -m src.train --trials 40          # train + tune + save model (≈ 2–4 min)
streamlit run app/streamlit_app.py       # dashboard  → http://localhost:8501
uvicorn api.main:app --reload            # REST API   → http://localhost:8000/docs
pytest                                   # 10 tests
```

## Dashboard tabs
- **🔬 Customer Lab**: build a customer profile (or load a preset or a random real customer) and get a live risk gauge, a verdict, SHAP reasons and retention actions.
- **📡 Batch Radar**: upload a CSV or scan the demo data. Shows KPIs (including monthly revenue at risk), the risk distribution, a contract × internet heat map, a searchable watchlist and a CSV export. You can open any customer in the Customer Lab.
- **🧠 Model Insights**: metrics, ROC curve, confusion matrix, global feature importance and the model leaderboard.
- **🗺️ Data Explorer**: churn rate by segment and numeric distributions for churned vs. stayed customers.

## API
| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness + model loaded |
| GET | `/model/info` | Model name, threshold, metrics, feature importance |
| POST | `/predict` | One customer (JSON) → probability, verdict, risk level, reasons, actions |
| POST | `/predict/batch` | CSV upload → predictions for every row |

## Project layout
```
src/        config, data cleaning, feature engineering, training, prediction, SHAP, retention playbook
app/        Streamlit dashboard (streamlit_app.py) + visual theme (style.py)
api/        FastAPI service
models/     churn_model.joblib + report.json (metrics, ROC, leaderboard)
notebooks/  EDA.ipynb
tests/      pytest suite
```

## Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io), click **Create app**, then **Deploy a public app from GitHub**.
2. Repository `Anirudh-Jakkani/churn-radar`, branch `main`, main file `app/streamlit_app.py`.
3. Under **Advanced settings**, pick Python 3.12 or newer, then click **Deploy**.

Cloud installs the lean `app/requirements.txt` (the dashboard doesn't need training or API packages). The trained model in `models/` ships with the repo.

## Docker
```bash
docker build -t churn .
docker run -p 8501:8501 churn
```
