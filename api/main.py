"""REST API for churn predictions.

Run:  uvicorn api.main:app --reload     (docs at http://localhost:8000/docs)
"""
import io
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.config import ID_COL
from src.predict import ChurnPredictor

YesNo = Literal["Yes", "No"]
AddOn = Literal["Yes", "No", "No internet service"]
state: dict = {}


class Customer(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: AddOn
    OnlineBackup: AddOn
    DeviceProtection: AddOn
    TechSupport: AddOn
    StreamingTV: AddOn
    StreamingMovies: AddOn
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {"json_schema_extra": {"example": {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 3,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.9, "TotalCharges": 269.7}}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictor"] = ChurnPredictor()
    yield
    state.clear()


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "predictor" in state}


@app.get("/model/info")
def model_info():
    r = state["predictor"].report
    return {k: r[k] for k in ("model_name", "trained_at", "threshold", "metrics", "feature_importance")}


@app.post("/predict")
def predict(customer: Customer, top_reasons: int = 5):
    result = state["predictor"].analyze(customer.model_dump())
    result["reasons"] = result["reasons"][:top_reasons]
    return result


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        preds = state["predictor"].predict(df)
    except (ValueError, pd.errors.ParserError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    if ID_COL in df:
        preds.insert(0, ID_COL, df[ID_COL])
    return {"n_customers": len(preds), "n_predicted_churn": int(preds["will_churn"].sum()),
            "predictions": preds.to_dict(orient="records")}
