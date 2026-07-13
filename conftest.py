import sys
import os
import pytest
import joblib
import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath('.'))


@pytest.fixture(autouse=True)
def create_test_model():
    os.makedirs('models', exist_ok=True)

    # create dummy training data
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(100, 30),
        columns=[f'feature_{i}' for i in range(30)]
    )
    y = np.random.randint(0, 2, 100)

    # build and train minimal pipeline
    pipeline = ImbPipeline([
        ('smote', BorderlineSMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis(n_components=1)),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X, y)

    # save dummy model
    joblib.dump(pipeline, 'models/churn_pipeline.pkl')

    # save dummy feature columns
    feature_columns = X.columns.tolist()
    joblib.dump(feature_columns, 'models/feature_columns.pkl')

    yield

    # cleanup after tests
    if os.path.exists('models/churn_pipeline.pkl'):
        os.remove('models/churn_pipeline.pkl')
    if os.path.exists('models/feature_columns.pkl'):
        os.remove('models/feature_columns.pkl')
