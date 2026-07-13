import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def clean_data(df):
    df = df.drop('customerID', axis=1)
    df['TotalCharges'] = df['TotalCharges'].replace(' ', '0.0')
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    df = df.replace({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
    df = df.astype({col: int for col in df.select_dtypes('bool').columns})
    return df


def encode_data(df):
    df = pd.get_dummies(df, drop_first=True)
    df = df.astype({col: int for col in df.select_dtypes('bool').columns})
    return df


def split_data(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test
