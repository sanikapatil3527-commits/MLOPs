from __future__ import annotations

from typing import List, Optional, Tuple
from mlflow.tracking import MlflowClient

from src.config import SETTINGS
from src.utils import (
    get_client,
    get_run_metric,
    get_current_production_version,
    transition_stage,
    tag_model_version,
    set_model_description_if_missing,
)


def list_versions(client: MlflowClient, model_name: str):
    return client.search_model_versions(f"name='{model_name}'")


def pick_best_candidate(
    client: MlflowClient,
    model_name: str,
    metric_name: str,
    exclude_stage: Optional[str] = None,
) -> Tuple[Optional[object], float]:
    """
    Choose the best model version by reading metrics from its run.
    exclude_stage can be 'Production' to avoid re-promoting current champion.
    """
    versions = list_versions(client, model_name)
    best_v = None
    best_score = -1.0

    for v in versions:
        stage = (v.current_stage or "").lower()
        if exclude_stage and stage == exclude_stage.lower():
            continue

        # Candidate pool: stage None or Staging is typical; we allow both
        run_id = v.run_id
        score = get_run_metric(client, run_id, metric_name)
        if score is None:
            continue

        if score > best_score:
            best_score = score
            best_v = v

    return best_v, best_score


def main():
    client = get_client(SETTINGS.tracking_uri)

    # nice description in Registry
    set_model_description_if_missing(
        client,
        SETTINGS.registered_model_name,
        "Day6 model registry demo: controlled promotion with metric thresholds and champion/challenger governance.",
    )

    # Current champion
    current_prod = get_current_production_version(client, SETTINGS.registered_model_name)
    if current_prod:
        champ_score = get_run_metric(client, current_prod.run_id, SETTINGS.primary_metric) or 0.0
        print(f"🏆 Current Production: v{current_prod.version} {SETTINGS.primary_metric}={champ_score:.4f}")
    else:
        champ_score = None
        print("ℹ️ No Production model yet.")

    # Best candidate among non-production
    candidate, cand_score = pick_best_candidate(
        client,
        SETTINGS.registered_model_name,
        SETTINGS.primary_metric,
        exclude_stage="Production",
    )

    if not candidate:
        raise RuntimeError("No candidate model versions found. Run training first.")

    print(f"🧪 Best Candidate: v{candidate.version} {SETTINGS.primary_metric}={cand_score:.4f} stage={candidate.current_stage}")

    # Always tag candidate evaluation for auditability
    tag_model_version(
        client,
        SETTINGS.registered_model_name,
        candidate.version,
        {
            "day": "Day6",
            "policy.primary_metric": SETTINGS.primary_metric,
            "policy.score": f"{cand_score:.6f}",
            "policy.status": "evaluated",
        },
    )

    # Promote to Staging (optional)
    if SETTINGS.promote_to_staging:
        if cand_score >= SETTINGS.min_primary_metric_for_staging:
            transition_stage(
                client,
                SETTINGS.registered_model_name,
                candidate.version,
                "Staging",
                archive_existing_versions=False,
            )
            tag_model_version(
                client,
                SETTINGS.registered_model_name,
                candidate.version,
                {"policy.stage": "Staging", "policy.status": "staging_pass"},
            )
            print(f"✅ Promoted to Staging: v{candidate.version}")
        else:
            print(f"❌ Staging gate failed: {cand_score:.4f} < {SETTINGS.min_primary_metric_for_staging}")
            return

    # Promote to Production (controlled)
    if SETTINGS.promote_to_production:
        if cand_score < SETTINGS.min_primary_metric_for_production:
            print(f"❌ Production gate failed: {cand_score:.4f} < {SETTINGS.min_primary_metric_for_production}")
            return

        # Optional: require beating the current champion
        if champ_score is not None and cand_score <= champ_score:
            print(f"❌ Not better than Champion: candidate={cand_score:.4f} <= champion={champ_score:.4f}")
            tag_model_version(
                client,
                SETTINGS.registered_model_name,
                candidate.version,
                {"policy.status": "rejected_not_better_than_champion"},
            )
            return

        # Promote candidate to Production and archive existing Production automatically
        transition_stage(
            client,
            SETTINGS.registered_model_name,
            candidate.version,
            "Production",
            archive_existing_versions=True,   # archives previous production
        )
        tag_model_version(
            client,
            SETTINGS.registered_model_name,
            candidate.version,
            {"policy.stage": "Production", "policy.status": "production_promoted", "role": "Champion"},
        )
        print(f"🚀 Promoted to Production (Champion): v{candidate.version}")

    print("🏁 Promotion finished. Check MLflow UI → Models.")


if __name__ == "__main__":
    main()