import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from src.config import SETTINGS


# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI(
    title="Churn Prediction API — Day 4",
    description="Production model served from MLflow Model Registry",
    version="1.0"
)

# -------------------------------------------------
# MLflow configuration
# -------------------------------------------------
mlflow.set_tracking_uri(SETTINGS.tracking_uri)

MODEL_URI = f"models:/{SETTINGS.registered_model_name}/Production"
_model = None


# -------------------------------------------------
# Input schema
# -------------------------------------------------
class ChurnInput(BaseModel):
    Age: float
    Total_Purchase: float
    Account_Manager: int
    Years: float
    Num_Sites: int
    Location: str
    Company: str


# -------------------------------------------------
# Model loading
# -------------------------------------------------
def load_model():
    global _model
    _model = mlflow.pyfunc.load_model(MODEL_URI)
    print(f"✅ Loaded Production model: {MODEL_URI}")


@app.on_event("startup")
def startup_event():
    try:
        load_model()
    except Exception as e:
        print(f"❌ Model load failed: {e}")


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/")
def root():
    if _model is None:
        return {
            "status": "error",
            "message": "No Production model available. Promote a model first."
        }

    return {
        "status": "ok",
        "model_uri": MODEL_URI
    }


@app.post("/predict")
def predict(items: List[ChurnInput]):
    if _model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. No Production model available."
        )

    df = pd.DataFrame([i.model_dump() for i in items])

    preds = _model.predict(df)

    return {
        "predictions": preds.tolist()
    }
