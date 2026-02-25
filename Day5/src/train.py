import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.config import SETTINGS


def get_models(seed: int = 42):
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, n_jobs=-1, random_state=seed))
        ]),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=seed),
    }


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def main():
    mlflow.set_tracking_uri(SETTINGS.tracking_uri)
    mlflow.set_experiment(SETTINGS.experiment_name)
    mlflow.end_run()

    df = pd.read_csv(SETTINGS.data_path)
    X = df.drop(columns=[SETTINGS.target_col])
    y = df[SETTINGS.target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = get_models(seed=42)

    # Save baseline distribution stats (for drift gate)
    baseline = {
        "columns": X_train.columns.tolist(),
        "train_means": X_train.mean().to_dict(),
        "train_stds": (X_train.std(ddof=0) + 1e-9).to_dict(),
    }

    # train loop
    for name, model in models.items():
        print(f"\n🚀 Training: {name}")
        mlflow.end_run()

        with mlflow.start_run(run_name=name):
            mlflow.set_tag("day", "Day5")
            mlflow.set_tag("pipeline", "CI-ML")
            mlflow.set_tag("dataset", "breast_cancer_wisconsin")
            mlflow.set_tag("model_name", name)

            # fit
            model.fit(X_train, y_train)

            # predict
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = y_pred.astype(float)

            metrics = compute_metrics(y_test, y_pred, y_proba)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # log artifacts needed for gates
            mlflow.log_dict(baseline, "baseline_stats.json")

            # log model
            mlflow.sklearn.log_model(model, artifact_path="model")

            print("✅ Metrics:", metrics)

    print("\n🏁 Training completed. Check MLflow runs in ./mlruns")

if __name__ == "__main__":
    main()
