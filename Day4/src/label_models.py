def label_models():
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(
        f"name='{SETTINGS.registered_model_name}'"
    )

    for v in versions:
        role = "challenger"
        if v.current_stage == "Production":
            role = "champion"

        client.set_model_version_tag(
            SETTINGS.registered_model_name,
            v.version,
            "role",
            role,
        )
