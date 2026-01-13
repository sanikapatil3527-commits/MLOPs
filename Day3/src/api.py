import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from src.config import SETTINGS

app = FastAPI(
    title="Churn Prediction API (Day 3)",
    version="1.0",
    description="Serves the PRODUCTION model from MLflow Model Registry"
)

# Connect to MLflow tracking server
mlflow.set_tracking_uri(SETTINGS.tracking_uri)

MODEL_URI = f"models:/{SETTINGS.registered_model_name}/Production"
_model = None

class ChurnInput(BaseModel):
    Age: float
    Total_Purchase: float
    Account_Manager: int
    Years: float
    Num_Sites: int
    Location: str
    Company: str

def load_model():
    global _model
    _model = mlflow.pyfunc.load_model(MODEL_URI)
    print(f"✅ Loaded model from Registry: {MODEL_URI}")

@app.on_event("startup")
def startup_event():
    try:
        load_model()
    except Exception as e:
        print(f"❌ Model load failed: {e}")

@app.get("/")
def root():
    if _model is None:
        return {"status": "error", "message": "Model not loaded. Is there a Production model?"}
    return {"status": "ok", "message": "API running", "model_uri": MODEL_URI}

@app.post("/predict")
def predict(items: List[ChurnInput]):
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Promote a model to Production first.")

    df = pd.DataFrame([i.model_dump() for i in items])

    preds = _model.predict(df)

    # Some pyfunc flavors may not expose predict_proba; keep robust
    probs = None
    if hasattr(_model, "predict_proba"):
        probs = _model.predict_proba(df)[:, 1].tolist()

    return {
        "predictions": preds.tolist(),
        "probabilities": probs
    }
