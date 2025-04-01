import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

def train_model_for_business(df, business_id, model_dir="models",model_type="random_forest"):
    # Preprocessing
    df = df.dropna()
    if 'CustomerID' in df.columns:
        df = df.drop(columns=["CustomerID"])
    if 'Churn' not in df.columns:
        raise ValueError("Missing target column: 'Churn'")

    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # One-hot encode
    X = pd.get_dummies(X, drop_first=False)

    # Balance classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Train model
    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
    else:
        raise ValueError("Unsupported model type")

    model.fit(X_resampled, y_resampled)

    # Save model + features
    model_path = f"{model_dir}/{business_id}"
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(model, f"{model_path}/churn_model.pkl")
    joblib.dump(list(X.columns), f"{model_path}/model_features.pkl")

    return {
        "model_path": model_path,
        "features": list(X.columns),
        "rows_used": len(X_resampled),
        "churn_rate": y.value_counts(normalize=True).to_dict()
    }
