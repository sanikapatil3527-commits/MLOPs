import mlflow
from src.config import SETTINGS

def main():
    mlflow.set_tracking_uri(SETTINGS.tracking_uri)
    client = mlflow.tracking.MlflowClient()

    exp = mlflow.get_experiment_by_name(SETTINGS.experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment '{SETTINGS.experiment_name}' not found. Train first.")

    # Get best run by F1
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.f1 DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs found. Train first.")

    best = runs[0]
    best_f1 = best.data.metrics.get("f1", None)
    if best_f1 is None:
        raise RuntimeError("Best run has no f1 metric logged.")

    print(f"🏆 Best run: {best.info.run_id} | f1={best_f1:.4f}")

    # Quality gate
    if best_f1 < SETTINGS.min_f1:
        raise ValueError(
            f"❌ Quality gate failed: best_f1={best_f1:.4f} < min_f1={SETTINGS.min_f1}"
        )

    print(f"✅ Quality gate passed: best_f1={best_f1:.4f} >= min_f1={SETTINGS.min_f1}")

if __name__ == "__main__":
    main()
