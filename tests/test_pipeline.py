from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

sample_customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'
    assert response.json()['model'] == 'loaded'


def test_predict_endpoint_returns_correct_fields():
    response = client.post('/predict', json=sample_customer)
    assert response.status_code == 200
    assert 'prediction' in response.json()
    assert 'churn_probability' in response.json()
    assert 'risk_level' in response.json()


def test_prediction_value_is_valid():
    response = client.post('/predict', json=sample_customer)
    prediction = response.json().get('prediction')
    assert prediction in ['Churned', 'Not Churned']


def test_risk_level_is_valid():
    response = client.post('/predict', json=sample_customer)
    risk = response.json().get('risk_level')
    assert risk in ['High Risk', 'Medium Risk', 'Low Risk']


def test_probability_is_between_0_and_1():
    response = client.post('/predict', json=sample_customer)
    probability = response.json().get('churn_probability')
    assert 0.0 <= probability <= 1.0