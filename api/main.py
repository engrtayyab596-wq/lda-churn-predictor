from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI(title='Customer Churn Predictor')

# load model and feature columns
model = joblib.load('models/churn_pipeline.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')

# optimal threshold from notebook
THRESHOLD = 0.5089

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model': 'loaded',
        'description': 'Customer Churn Predictor API'
    }


@app.post('/predict')
def predict(data: CustomerData):
    # step 1 - convert to dataframe
    raw = pd.DataFrame([data.dict()])

    # step 2 - label encode binary columns
    raw = raw.replace({'Yes': 1, 'No': 0,
                       'Male': 1, 'Female': 0})

    # step 3 - one hot encode
    raw = pd.get_dummies(raw)

    # step 4 - align columns with training data
    raw = raw.reindex(columns=feature_columns, fill_value=0)

    # step 5 - get probability
    probability = model.predict_proba(raw)[0][1]

    # step 6 - apply optimal threshold
    prediction = 1 if probability >= THRESHOLD else 0

    # step 7 - return result
    result = 'Churned' if prediction == 1 else 'Not Churned'

    # step 8 - add risk level
    if probability >= 0.75:
        risk = 'High Risk'
    elif probability >= 0.50:
        risk = 'Medium Risk'
    else:
        risk = 'Low Risk'

    return {
        'prediction': result,
        'churn_probability': round(float(probability), 4),
        'risk_level': risk
    }
