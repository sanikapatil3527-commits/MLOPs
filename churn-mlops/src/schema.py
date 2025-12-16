from pydantic import BaseModel
from typing import Dict, Any

class PredictRequest(BaseModel):
    # On accepte un dict flexible (pratique en cours)
    features: Dict[str, Any]
