# Start MLflow Tracking Server with SQLite backend (Registry enabled)
# Run from repo root: powershell -ExecutionPolicy Bypass -File scripts/start_mlflow_server.ps1

$ErrorActionPreference = "Stop"

mlflow server `
  --host 127.0.0.1 `
  --port 5000 `
  --backend-store-uri sqlite:///mlflow.db `
  --default-artifact-root ./mlruns
