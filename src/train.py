import joblib
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def build_pipeline():
    pipeline = ImbPipeline([
        ('smote', BorderlineSMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis(n_components=1)),
        ('classifier', LogisticRegression(
            max_iter=1000, C=0.01, solver='lbfgs'
        ))
    ])
    return pipeline


def train_model(pipeline, X_train, y_train):
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=5, scoring='f1'
    )
    print(f"Mean F1: {cv_scores.mean():.4f}")
    print(f"Std F1:  {cv_scores.std():.4f}")
    pipeline.fit(X_train, y_train)
    return pipeline, cv_scores


def save_model(pipeline, path='models/churn_pipeline.pkl'):
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")
