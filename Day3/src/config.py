from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    tracking_uri: str = "http://127.0.0.1:5000"
    experiment_name: str = "churn-day3"
    registered_model_name: str = "ChurnClassifier"
    target_col: str = "Churn"
    data_path: str = "data/churn.csv"

SETTINGS = Settings()
