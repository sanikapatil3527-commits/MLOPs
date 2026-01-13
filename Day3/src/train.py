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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.config import SETTINGS

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ]
    )

def get_models(seed: int = 42):
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=250, random_state=seed),
    }

def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=SETTINGS.data_path)
    parser.add_argument("--target", default=SETTINGS.target_col)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Connect to tracking server (Registry-capable)
    mlflow.set_tracking_uri(SETTINGS.tracking_uri)
    mlflow.set_experiment(SETTINGS.experiment_name)

    # Safety: close any ghost run
    mlflow.end_run()

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found. Columns: {list(df.columns)}")

    X = df.drop(columns=[args.target])
    y = df[args.target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    preprocessor = build_preprocessor(X)
    models = get_models(seed=args.seed)

    for model_name, model in models.items():
        print(f"\n🚀 Training model: {model_name}")

        mlflow.end_run()  # make sure no run is active
        run = mlflow.start_run(run_name=model_name)

        try:
            # Tags / metadata (Day2 objective)
            mlflow.set_tag("project", "PSTB_MLOps")
            mlflow.set_tag("day", "day3")
            mlflow.set_tag("business_goal", "churn_reduction")
            mlflow.set_tag("model_role", "candidate")
            mlflow.set_tag("model_name", model_name)

            mlflow.log_param("test_size", args.test_size)
            mlflow.log_param("seed", args.seed)

            pipe = Pipeline(steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ])

            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]

            metrics = compute_metrics(y_test, y_pred, y_proba)
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            # Log model artifact
            mlflow.sklearn.log_model(pipe, artifact_path="model")

            print("✅ Metrics:")
            for k, v in metrics.items():
                print(f"   {k}: {v:.4f}")

        finally:
            mlflow.end_run()

if __name__ == "__main__":
    main()
