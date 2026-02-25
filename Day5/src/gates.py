#We’ll compute PSI (Population Stability Index) on numeric features.

#PSI small → stable

#PSI large → drift

import json
import numpy as np
import pandas as pd
import mlflow

from src.config import SETTINGS


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Simple PSI for numeric vectors."""
    expected = expected.astype(float)
    actual = actual.astype(float)

    # Build bins based on expected quantiles
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    if len(cuts) < 3:
        return 0.0  # degenerate feature

    exp_counts, _ = np.histogram(expected, bins=cuts)
    act_counts, _ = np.histogram(actual, bins=cuts)

    exp_perc = exp_counts / (exp_counts.sum() + 1e-9)
    act_perc = act_counts / (act_counts.sum() + 1e-9)

    exp_perc = np.clip(exp_perc, 1e-6, 1)
    act_perc = np.clip(act_perc, 1e-6, 1)

    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))


def main():
    mlflow.set_tracking_uri(SETTINGS.tracking_uri)
    client = mlflow.tracking.MlflowClient()

    exp = mlflow.get_experiment_by_name(SETTINGS.experiment_name)
    if exp is None:
        raise RuntimeError("Experiment not found. Train first.")

    # Use best run (by f1) as baseline
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.f1 DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs found. Train first.")

    run = runs[0]
    run_id = run.info.run_id

    # Load baseline stats artifact
    local_path = client.download_artifacts(run_id, "baseline_stats.json")
    with open(local_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    df = pd.read_csv(SETTINGS.data_path)
    X = df.drop(columns=[SETTINGS.target_col])

    # Simulate drift (for teaching): shift a few columns
    X_drift = X.copy()
    cols = baseline["columns"][:3]  # first 3 features
    for c in cols:
        X_drift[c] = X_drift[c] * 1.15  # 15% shift

    # Compute PSI feature-wise
    drift_report = {}
    max_psi = 0.0
    for c in baseline["columns"]:
        p = psi(X[c].values, X_drift[c].values, bins=10)
        drift_report[c] = p
        max_psi = max(max_psi, p)

    print(f"📊 Drift check (max PSI): {max_psi:.4f}")

    if max_psi > SETTINGS.max_drift_psi:
        raise RuntimeError(
            f"❌ Drift gate failed: max_psi={max_psi:.4f} > threshold={SETTINGS.max_drift_psi}"
        )

    print(f"✅ Drift gate passed: max_psi={max_psi:.4f} <= threshold={SETTINGS.max_drift_psi}")

if __name__ == "__main__":
    main()
