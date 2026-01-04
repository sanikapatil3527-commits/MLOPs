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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# import mlflow
# import os

# mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))


def build_pipeline(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols)
        ],
        remainder="drop"
    )

    model = LogisticRegression(max_iter=2000)

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", model)
    ])

    return pipe, X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/churn.csv", help="Path to churn CSV")
    parser.add_argument("--target", default="churn", help="Target column name")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0, help="LogReg inverse regularization strength")
    parser.add_argument("--experiment", default="churn-mlops", help="MLflow experiment name")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found. Columns: {list(df.columns)}")

    pipe, X, y = build_pipeline(df, args.target)

    # Inject hyperparam
    pipe.set_params(clf__C=args.C)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        # Params
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", args.C)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        # Train
        pipe.fit(X_train, y_train)

        # Predict
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        # Metrics
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        auc = roc_auc_score(y_test, proba)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", auc)

        # Report artifact
        report = classification_report(y_test, pred)
        os.makedirs("models", exist_ok=True)
        report_path = os.path.join("models", "classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # Model artifact
        # Signature / input example helps later, but keep simple for class
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model"
        )

        print("✅ Training done.")
        print(f"accuracy={acc:.3f} | f1={f1:.3f} | roc_auc={auc:.3f}")
        print("Run logged to MLflow.")


if __name__ == "__main__":
    main()
