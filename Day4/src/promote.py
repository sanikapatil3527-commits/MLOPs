import mlflow
from src.config import SETTINGS

def promote_to_staging(metric="f1", threshold=0.75):
    mlflow.set_tracking_uri(SETTINGS.tracking_uri)
    client = mlflow.tracking.MlflowClient()

    exp = mlflow.get_experiment_by_name(SETTINGS.experiment_name)
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )

    best = runs[0]
    score = best.data.metrics[metric]

    if score < threshold:
        raise RuntimeError("Quality gate failed")

    model_uri = f"runs:/{best.info.run_id}/model"
    result = mlflow.register_model(model_uri, SETTINGS.registered_model_name)

    client.transition_model_version_stage(
        name=SETTINGS.registered_model_name,
        version=result.version,
        stage="Staging",
        archive_existing_versions=False,
    )

    print(f"Model promoted to STAGING (v{result.version})")
