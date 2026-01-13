import mlflow
from mlflow.exceptions import MlflowException
from src.config import SETTINGS

def get_best_candidate_run_id(metric: str = "f1") -> str:
    mlflow.set_tracking_uri(SETTINGS.tracking_uri)
    exp = mlflow.get_experiment_by_name(SETTINGS.experiment_name)
    if exp is None:
        raise RuntimeError("Experiment not found. Train models first.")

    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.model_role = 'candidate'",
        order_by=[f"metrics.{metric} DESC"],
        max_results=50,
    )
    if not runs:
        raise RuntimeError("No candidate runs available.")
    return runs[0].info.run_id

def register_best_model(model_name: str = SETTINGS.registered_model_name) -> int:
    """
    Registers the best candidate run model to MLflow Model Registry.
    Returns the created model version.
    """
    run_id = get_best_candidate_run_id(metric="f1")
    model_uri = f"runs:/{run_id}/model"

    mlflow.set_tracking_uri(SETTINGS.tracking_uri)

    # Register model
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"✅ Registered model: {model_name} v{result.version} (run_id={run_id})")
    return int(result.version)

def transition_stage(model_name: str, version: int, stage: str):
    """
    Transitions a model version to a stage: Staging, Production, Archived.
    """
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )
    print(f"✅ Transitioned {model_name} v{version} → {stage}")

def annotate_version(model_name: str, version: int, description: str):
    client = mlflow.tracking.MlflowClient()
    client.update_model_version(
        name=model_name,
        version=str(version),
        description=description
    )
    print(f"📝 Updated description for {model_name} v{version}")

if __name__ == "__main__":
    # Example flow (manual run)
    v = register_best_model()
    annotate_version(
        SETTINGS.registered_model_name,
        v,
        "Day3: best candidate selected by F1. Ready for QA in Staging."
    )
    transition_stage(SETTINGS.registered_model_name, v, "Staging")
