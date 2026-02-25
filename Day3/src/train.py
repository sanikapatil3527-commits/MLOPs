import argparse
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.config import SETTINGS


# -------------------------------------------------
# Preprocessing
# -------------------------------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ]
    )


# -------------------------------------------------
# Models
# -------------------------------------------------
def get_models(seed: int = 42):
    """
    A diverse and MLOps-safe model zoo:
    - baseline
    - interpretable
    - robust ensemble
    - high-performance booster
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            n_jobs=-1
        ),

        "DecisionTree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=50,
            random_state=seed
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=20,
            random_state=seed,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1
        ),
    }


# -------------------------------------------------
# Metrics
# -------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=SETTINGS.data_path)
    parser.add_argument("--target", default=SETTINGS.target_col)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- MLflow configuration (Day 3: service mode)
    mlflow.set_tracking_uri(SETTINGS.tracking_uri)
    mlflow.set_experiment(SETTINGS.experiment_name)

    # Safety: close any ghost run
    mlflow.end_run()

    # --- Load data
    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(
            f"Target '{args.target}' not found. Columns: {list(df.columns)}"
        )

    X = df.drop(columns=[args.target])
    y = df[args.target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    preprocessor = build_preprocessor(X)
    models = get_models(seed=args.seed)

    # -------------------------------------------------
    # Train loop (one MLflow run per model)
    # -------------------------------------------------
    for model_name, model in models.items():
        print(f"\n🚀 Training model: {model_name}")

        mlflow.end_run()
        with mlflow.start_run(run_name=model_name):

            # -------- Tags (MLOps metadata)
            mlflow.set_tag("project", "PSTB_MLOps")
            mlflow.set_tag("course_day", "Day3")
            mlflow.set_tag("business_goal", "customer_churn")
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("model_role", "candidate")

            if model_name in ["LogisticRegression", "DecisionTree"]:
                mlflow.set_tag("interpretability", "high")
            else:
                mlflow.set_tag("interpretability", "low")

            # -------- Parameters
            mlflow.log_param("test_size", args.test_size)
            mlflow.log_param("seed", args.seed)

            # -------- Pipeline
            pipeline = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", model),
                ]
            )

            pipeline.fit(X_train, y_train)

            # -------- Evaluation
            y_pred = pipeline.predict(X_test)

            if hasattr(pipeline, "predict_proba"):
                y_proba = pipeline.predict_proba(X_test)[:, 1]
            else:
                # fallback (rare)
                y_proba = y_pred

            metrics = compute_metrics(y_test, y_pred, y_proba)

            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            # -------- Model artifact
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                registered_model_name=None  # registry handled in registry.py
            )

            print("✅ Metrics:")
            for k, v in metrics.items():
                print(f"   {k}: {v:.4f}")

    print("\n🏁 Training completed for all candidate models.")


if __name__ == "__main__":
    main()
