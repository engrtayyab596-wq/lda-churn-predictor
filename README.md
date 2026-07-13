# Customer Churn Predictor

A production-grade machine learning system that predicts customer churn
using Linear Discriminant Analysis (LDA) and Logistic Regression,
served via a REST API with optimal threshold tuning.

---

## Project Overview

This project demonstrates an end-to-end ML engineering pipeline built on the
IBM Telco Customer Churn dataset. It covers the full stack from raw data
exploration to a containerised, tested, and automatically verified API service.

The project addresses a real business problem — identifying customers likely
to cancel their subscription so retention teams can intervene proactively.

---

## Results

| Metric | Score |
|---|---|
| Cross-validation F1 | 0.6059 |
| Final F1 (with threshold tuning) | 0.6436 |
| Recall on churned customers | 72.39% |
| Precision | 57.94% |
| Optimal threshold | 0.5089 |
| LDA components | 1 (binary classification) |

---

## Model Comparison

| Model | F1 | Recall | Precision |
|---|---|---|---|
| Logistic Regression | 0.6436 | 0.7239 | 0.5794 |
| SVM | 0.6436 | 0.7802 | 0.5310 |
| XGBoost | 0.6366 | 0.7507 | 0.5523 |
| Random Forest | 0.5630 | 0.9732 | 0.3630 |

Logistic Regression was selected as the final model — highest F1 with
best precision-recall balance for a retention team use case.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Machine Learning | scikit-learn, LDA, Logistic Regression |
| Imbalanced Data | imbalanced-learn, BorderlineSMOTE |
| Data Analysis | pandas, numpy, seaborn, matplotlib |
| Experiment Tracking | MLflow |
| API | FastAPI, uvicorn, pydantic |
| Testing | pytest |
| Containerisation | Docker |
| CI/CD | GitHub Actions |

---

## Project Structure

```
lda-churn-predictor/
├── data/                        
├── notebooks/                   
│   └── 01_eda_and_lda.ipynb
├── src/                         
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── visualize.py
├── api/                         
│   └── main.py
├── models/                      
├── tests/                       
│   └── test_pipeline.py
├── Dockerfile
├── requirements.txt
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Why LDA?

Linear Discriminant Analysis was chosen over PCA for this classification
problem because LDA is a supervised dimensionality reduction technique.
It uses class labels to find the direction that maximally separates
churned from non-churned customers. For binary classification, LDA
compresses all 30 features into 1 highly discriminative dimension.

---

## Why BorderlineSMOTE?

The dataset has significant class imbalance — 73.5% not churned vs
26.5% churned. BorderlineSMOTE was chosen over simple SMOTE because
churn data has significant class overlap near the decision boundary.
BorderlineSMOTE focuses synthetic sample generation exactly at the
boundary where the model struggles most, improving recall on the
minority churned class.

---

## Why Threshold Tuning?

The default classification threshold of 0.5 was not optimal for this
imbalanced dataset. By computing the precision-recall curve and finding
the threshold that maximises F1, an optimal threshold of 0.5089 was
identified — improving F1 from baseline to 0.6436 and recall to 72.39%.

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/lda-churn-predictor.git
cd lda-churn-predictor
```

### 2. Create conda environment

```bash
conda create -n ML_P python=3.11
conda activate ML_P
pip install -r requirements.txt
```

### 3. Add the dataset

Download the Telco Customer Churn dataset from Kaggle and place it at:

```
data/churn_data.csv
```

### 4. Generate the model

Open and run all cells in notebooks/01_eda_and_lda.ipynb.
This trains the pipeline and saves it to models/churn_pipeline.pkl.

### 5. Start the API

```bash
uvicorn api.main:app --reload
```

### 6. Open the interactive docs

```
http://127.0.0.1:8000/docs
```

---

## How to Run with Docker

```bash
docker build -t churn-predictor .
docker run -p 8000:8000 churn-predictor
```

---

## How to Run Tests

```bash
pytest tests/ -v
```

---

## Experiment Tracking

This project uses MLflow for experiment tracking. All four models were
logged with parameters, metrics and artifacts for comparison.

To view experiments locally:

```bash
cd notebooks
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open: http://127.0.0.1:5000

---

## API Endpoints

### GET /health

Returns API status.

```json
{
  "status": "ok",
  "model": "loaded",
  "description": "Customer Churn Predictor API"
}
```

### POST /predict

Accepts raw customer data and returns churn prediction.

Example request:

```json
{
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
```

Example response:

```json
{
  "prediction": "Churned",
  "churn_probability": 0.7823,
  "risk_level": "High Risk"
}
```

---

## Dataset

IBM Telco Customer Churn Dataset

- 7043 customers
- 19 features covering demographic, account and service data
- Binary classification: Churned (26.5%) vs Not Churned (73.5%)
- Mixed data types: numerical and categorical

---

## Key Design Decisions

- BorderlineSMOTE inside ImbPipeline prevents data leakage during CV
- SMOTE applied only to training folds — test set reflects real distribution
- Threshold tuned using precision-recall curve — not default 0.5
- joblib used for model serialisation — Docker compatible
- Four classifiers compared with evidence — Logistic Regression selected
- MLflow tracks all experiment runs for reproducibility

---

## Author

Tayyab
ML/AI Engineering Portfolio Project