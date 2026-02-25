# Start an MLflow server (backend store + artifacts) for Registry
# From Day6/ root:
#   powershell -ExecutionPolicy Bypass -File scripts/start_mlflow_server.ps1

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path "mlflow" | Out-Null
New-Item -ItemType Directory -Force -Path "mlflow\artifacts" | Out-Null

mlflow server `
  --host 127.0.0.1 `
  --port 5000 `
  --backend-store-uri sqlite:///mlflow/mlflow.db `
  --default-artifact-root ./mlflow/artifacts