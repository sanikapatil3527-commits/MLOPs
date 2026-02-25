from __future__ import annotations

from typing import Dict, Optional, Tuple
import mlflow
from mlflow.tracking import MlflowClient


def set_mlflow(tracking_uri: str, experiment_name: str) -> str:
    """Configure MLflow tracking + ensure experiment exists. Returns experiment_id."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.end_run()
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(experiment_name)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_name)
    return exp_id


def get_client(tracking_uri: str) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def get_run_metric(client: MlflowClient, run_id: str, metric_name: str) -> Optional[float]:
    run = client.get_run(run_id)
    m = run.data.metrics.get(metric_name)
    return float(m) if m is not None else None


def get_current_production_version(client: MlflowClient, model_name: str):
    """Return the current Production model version (or None)."""
    # search_model_versions returns objects with current_stage
    versions = client.search_model_versions(f"name='{model_name}'")
    prod = [v for v in versions if (v.current_stage or "").lower() == "production"]
    if not prod:
        return None
    # if multiple (shouldn't), pick latest by version number
    prod.sort(key=lambda v: int(v.version), reverse=True)
    return prod[0]


def transition_stage(
    client: MlflowClient,
    model_name: str,
    version: str,
    stage: str,
    archive_existing_versions: bool = False,
) -> None:
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=stage,
        archive_existing_versions=archive_existing_versions,
    )


def tag_model_version(
    client: MlflowClient,
    model_name: str,
    version: str,
    tags: Dict[str, str],
) -> None:
    for k, v in tags.items():
        client.set_model_version_tag(name=model_name, version=str(version), key=k, value=str(v))


def set_model_description_if_missing(client: MlflowClient, model_name: str, text: str) -> None:
    try:
        model = client.get_registered_model(model_name)
        if not model.description:
            client.update_registered_model(name=model_name, description=text)
    except Exception:
        # model may not exist yet
        pass