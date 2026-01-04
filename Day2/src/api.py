"""
API of prediction – Day 2 MLOps
--------------------------------
- Charge automatically the model CHAMPION from MLflow
- Expose an API REST via FastAPI
- Prepare  transition towards MLflow Model Registry (Day 3)

Launch:
    uvicorn src.api:app --reload
"""

import pathlib
import mlflow
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List


# ============================================================
#  MLflow configuration (portable & safe)
# ============================================================

MLFLOW_TRACKING_URI = pathlib.Path("mlruns").absolute().as_uri()
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "churn-day2"


# ============================================================
#  FastAPI app
# ============================================================

app = FastAPI(
    title="Churn Prediction API",
    description="Day 2 MLOps – Champion model serving",
    version="1.0"
)


# ============================================================
#  Input schema
# ============================================================

class ChurnInput(BaseModel):
    Age: float
    Total_Purchase: float
    Account_Manager: int
    Years: float
    Num_Sites: int
    Location: str
    Company: str


# ============================================================
#  Model loading logic
# ============================================================

def load_champion_model():
    """
    Load the champion model from MLflow based on tags.
    Champion = run tagged with model_role=champion
    """
    client = mlflow.tracking.MlflowClient()

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT_NAME}' not found. "
            "Train models before starting the API."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.model_role = 'champion'",
        order_by=["metrics.f1 DESC"]
    )

    if not runs:
        raise RuntimeError(
            "No champion model found. "
            "Run select_champion.py first."
        )

    champion_run = runs[0]
    model_uri = f"runs:/{champion_run.info.run_id}/model"

    print(f"🏆 Loading champion model from run_id: {champion_run.info.run_id}")
    return mlflow.pyfunc.load_model(model_uri)


# ============================================================
#  Load model at startup
# ============================================================

try:
    model = load_champion_model()
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None


# ============================================================
#  Health check
# ============================================================

@app.get("/")
def root():
    if model is None:
        return {
            "status": "error",
            "message": "Model not loaded"
        }
    return {
        "status": "ok",
        "message": "Churn Prediction API is running",
        "model": "champion"
    }


# ============================================================
# 🔮 Prediction endpoint
# ============================================================

@app.post("/predict")
def predict(data: List[ChurnInput]):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not available"
        )

    df = pd.DataFrame([d.dict() for d in data])

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    return {
        "predictions": preds.tolist(),
        "probabilities": probs.tolist()
    }



