
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


VALID_CUSTOMER = {
    "LIMIT_BAL": 50000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 30,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 45000,
    "BILL_AMT2": 42000,
    "BILL_AMT3": 40000,
    "BILL_AMT4": 38000,
    "BILL_AMT5": 35000,
    "BILL_AMT6": 30000,
    "PAY_AMT1": 0,
    "PAY_AMT2": 1000,
    "PAY_AMT3": 2000,
    "PAY_AMT4": 2000,
    "PAY_AMT5": 3000,
    "PAY_AMT6": 3000,
}


def test_health_endpoint():
    with TestClient(app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
    }


def test_predict_valid_customer():
    with TestClient(app) as test_client:
        response = test_client.post(
            "/predict",
            json=VALID_CUSTOMER,
        )

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in (0, 1)
    assert result["prediction_label"] in ("Default", "No Default")
    assert 0.0 <= result["default_probability"] <= 1.0


def test_predict_rejects_missing_feature():
    customer = VALID_CUSTOMER.copy()
    customer.pop("LIMIT_BAL")

    with TestClient(app) as test_client:
        response = test_client.post("/predict", json=customer)

    assert response.status_code == 422


def test_predict_rejects_unexpected_feature():
    customer = VALID_CUSTOMER.copy()
    customer["unexpected_feature"] = 123

    with TestClient(app) as test_client:
        response = test_client.post("/predict", json=customer)

    assert response.status_code == 422
