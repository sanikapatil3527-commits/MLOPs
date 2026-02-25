from __future__ import annotations

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.config import SETTINGS
from src.utils import set_mlflow


def get_models(seed: int = 42):
    return {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, n_jobs=-1, random_state=seed)),
        ]),
        "RandomForest": RandomForestClassifier(n_estimators=350, random_state=seed, n_jobs=-1),
        "GradBoost": GradientBoostingClassifier(random_state=seed),
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
    set_mlflow(SETTINGS.tracking_uri, SETTINGS.experiment_name)

    df = pd.read_csv(SETTINGS.data_path)
    X = df.drop(columns=[SETTINGS.target_col])
    y = df[SETTINGS.target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    for model_name, model in get_models().items():
        mlflow.end_run()
        with mlflow.start_run(run_name=model_name):
            mlflow.set_tag("day", "Day6")
            mlflow.set_tag("registered_model_name", SETTINGS.registered_model_name)
            mlflow.set_tag("candidate", "true")
            mlflow.set_tag("model_family", model_name)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

            metrics = compute_metrics(y_test, y_pred, y_proba)
            mlflow.log_metrics(metrics)

            # Register in Model Registry
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=SETTINGS.registered_model_name,
            )

            print(f"✅ {model_name} metrics={metrics}")

    print("\n🏁 Done. Open MLflow UI → Models →", SETTINGS.registered_model_name)


if __name__ == "__main__":
    main()