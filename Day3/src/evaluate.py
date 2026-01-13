import mlflow
from src.config import SETTINGS

def main():
    mlflow.set_tracking_uri(SETTINGS.tracking_uri)

    exp = mlflow.get_experiment_by_name(SETTINGS.experiment_name)
    if exp is None:
        raise RuntimeError("Experiment not found. Run training first.")

    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.model_role = 'candidate'",
        order_by=["metrics.f1 DESC"],
        max_results=20,
    )
    if not runs:
        raise RuntimeError("No candidate runs found.")

    best = runs[0]
    print("🏆 Best candidate run:")
    print(f"   run_id: {best.info.run_id}")
    print(f"   model_name: {best.data.tags.get('model_name')}")
    print(f"   f1: {best.data.metrics.get('f1')}")

if __name__ == "__main__":
    main()
