import pandas as pd

from src.data import clean
from src.features import ENGINEERED_CATEGORICAL, ENGINEERED_NUMERIC, add_features, original_feature
from src.retention import recommend

CUSTOMER = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 0,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.9, "TotalCharges": " ",
}


def test_clean_fills_blank_total_charges():
    df = clean(pd.DataFrame([CUSTOMER]))
    assert df.loc[0, "TotalCharges"] == 0


def test_add_features_creates_engineered_columns():
    X = add_features(clean(pd.DataFrame([CUSTOMER])))
    for col in ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL:
        assert col in X
    assert X.loc[0, "tenure_bucket"] == "0-6m"
    assert X.loc[0, "auto_pay"] == "No"
    assert X.loc[0, "num_services"] == 2  # phone + streaming TV


def test_original_feature_mapping():
    assert original_feature("num__tenure") == "tenure"
    assert original_feature("cat__Contract_Two year") == "Contract"
    assert original_feature("cat__tenure_bucket_0-6m") == "tenure_bucket"
    assert original_feature("cat__StreamingTV_No internet service") == "StreamingTV"


def test_recommend_only_uses_churn_drivers():
    feats = add_features(clean(pd.DataFrame([CUSTOMER]))).iloc[0].to_dict()
    reasons = [{"feature": "Contract", "impact": 0.8}, {"feature": "PaymentMethod", "impact": -0.2}]
    titles = [a["title"] for a in recommend(feats, reasons)]
    assert titles == ["📝 Lock-in offer"]


def test_recommend_fallback_when_no_drivers():
    feats = add_features(clean(pd.DataFrame([CUSTOMER]))).iloc[0].to_dict()
    assert recommend(feats, [{"feature": "Contract", "impact": -0.5}])[0]["title"].startswith("✅")
