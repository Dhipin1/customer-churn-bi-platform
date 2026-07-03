from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()

def test_predict_schema():
    # Requires trained model first:
    # python -m src.train
    sample = {
        "tenure": 1,
        "MonthlyCharges": 70,
        "TotalCharges": 70,
        "SeniorCitizen": 0,
        "gender": "Male",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }
    r = client.post("/api/predict", json={"data": sample})
    assert r.status_code == 200
    body = r.json()
    assert "churn_probability" in body
    assert "model_version" in body