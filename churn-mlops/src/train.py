import os
import argparse
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ------------------------------------------------------------------
# 🔒 Force MLflow to be local & portable (VERY IMPORTANT for students)
# ------------------------------------------------------------------
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))


# ------------------------------------------------------------------
# Build preprocessing + split
# ------------------------------------------------------------------
def prepare_data(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols)
        ]
    )

    return X, y, preprocessor


# ------------------------------------------------------------------
# Define models to compare
# ------------------------------------------------------------------
def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42
        )
    }


# ------------------------------------------------------------------
# Metrics computation
# ------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/churn.csv")
    parser.add_argument("--target", default="Churn")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--experiment", default="churn-multi-model")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise ValueError(
            f"Target '{args.target}' not found. Available columns: {list(df.columns)}"
        )

    X, y, preprocessor = prepare_data(df, args.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y
    )

    mlflow.set_experiment(args.experiment)

    models = get_models()

    for model_name, model in models.items():
        print(f"\n🚀 Training model: {model_name}")

        pipeline = Pipeline(steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        with mlflow.start_run(run_name=model_name):
            # Log model info
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("test_size", args.test_size)

            # Train
            pipeline.fit(X_train, y_train)

            # Predict
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            # Metrics
            metrics = compute_metrics(y_test, y_pred, y_proba)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log model
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model"
            )

            print(f"✅ {model_name} metrics:")
            for k, v in metrics.items():
                print(f"   {k}: {v:.3f}")


if __name__ == "__main__":
    main()
