from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    # Data
    data_path: str = "data/breast_cancer.csv"
    target_col: str = "label"

    # MLflow server (Registry requires a server backend store)
    tracking_uri: str = "http://127.0.0.1:5000"
    experiment_name: str = "day6-registry"

    # Registry model name
    registered_model_name: str = "ClassifierDay6"

    # Promotion policy
    primary_metric: str = "roc_auc"     # can be "f1" too
    promote_to_staging: bool = True
    promote_to_production: bool = True

    # Guardrails
    min_primary_metric_for_staging: float = 0.90
    min_primary_metric_for_production: float = 0.92

SETTINGS = Settings()