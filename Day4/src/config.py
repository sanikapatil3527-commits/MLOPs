from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # MLflow
    tracking_uri: str = "http://127.0.0.1:5000"
    experiment_name: str = "churn-day3"   # continuity with Day3
    registered_model_name: str = "ChurnClassifier"

    # Data
    data_path: str = "data/churn.csv"
    target_col: str = "Churn"

    # CD/ML thresholds
    promotion_metric: str = "f1"
    promotion_threshold: float = 0.75


SETTINGS = Settings()
