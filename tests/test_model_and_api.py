import io

import pytest

from src.config import MODEL_PATH, RAW_DATA
from tests.test_features import CUSTOMER

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="train the model first")

LOYAL = {**CUSTOMER, "tenure": 70, "Contract": "Two year", "PaymentMethod": "Credit card (automatic)",
         "InternetService": "DSL", "OnlineSecurity": "Yes", "TechSupport": "Yes",
         "MonthlyCharges": 60.0, "TotalCharges": 4200.0}
RISKY = {**CUSTOMER, "tenure": 2, "TotalCharges": 179.8}


@pytest.fixture(scope="module")
def predictor():
    from src.predict import ChurnPredictor
    return ChurnPredictor()


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_probabilities_are_valid_and_ordered(predictor):
    loyal = predictor.analyze(LOYAL)
    risky = predictor.analyze(RISKY)
    assert 0 <= loyal["churn_probability"] <= 1
    assert risky["churn_probability"] > loyal["churn_probability"]
    assert risky["will_churn"] and not loyal["will_churn"]


def test_explanations_cover_features(predictor):
    reasons = predictor.analyze(RISKY)["reasons"]
    names = {r["feature"] for r in reasons}
    assert {"Contract", "tenure", "InternetService"} <= names
    impacts = [abs(r["impact"]) for r in reasons]
    assert impacts == sorted(impacts, reverse=True)


def test_api_health_and_predict(client):
    assert client.get("/health").json()["model_loaded"] is True
    body = client.post("/predict", json=RISKY).json()
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert len(body["reasons"]) == 5
    assert body["recommended_actions"]


def test_api_rejects_bad_input(client):
    assert client.post("/predict", json={**RISKY, "Contract": "Forever"}).status_code == 422


def test_api_batch(client):
    sample = "".join(RAW_DATA.read_text().splitlines(keepends=True)[:51])
    r = client.post("/predict/batch", files={"file": ("c.csv", io.BytesIO(sample.encode()), "text/csv")})
    assert r.status_code == 200
    assert r.json()["n_customers"] == 50
