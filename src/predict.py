import joblib
import pandas as pd

THRESHOLD = 0.5089


def load_model(model_path='models/churn_pipeline.pkl',
               columns_path='models/feature_columns.pkl'):
    model = joblib.load(model_path)
    feature_columns = joblib.load(columns_path)
    return model, feature_columns


def preprocess_input(data: dict, feature_columns: list):
    raw = pd.DataFrame([data])
    raw = raw.replace({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
    raw = raw.infer_objects(copy=False)
    raw = pd.get_dummies(raw)
    raw = raw.reindex(columns=feature_columns, fill_value=0)
    return raw


def predict(model, feature_columns, data: dict):
    raw = preprocess_input(data, feature_columns)
    probability = model.predict_proba(raw)[0][1]
    prediction = 1 if probability >= THRESHOLD else 0
    result = 'Churned' if prediction == 1 else 'Not Churned'
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
