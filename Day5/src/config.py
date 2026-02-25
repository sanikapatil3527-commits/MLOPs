from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    data_path: str = "data/breast_cancer.csv"
    target_col: str = "label"

    # MLflow local tracking (no server required)
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "breast-cancer-day5"

    # Quality gates (tune if needed)
    min_f1: float = 0.92
    max_drift_psi: float = 0.20  # per-feature PSI threshold

SETTINGS = Settings()
