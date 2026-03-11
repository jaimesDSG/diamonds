from enum import Enum
from typing import Annotated
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from diamonds.registry import load_model

app = FastAPI()
model = load_model(path="models")

# 1. On définit les choix possibles pour les catégories
class CutEnum(str, Enum):
    fair = "Fair"
    good = "Good"
    very_good = "Very Good"
    premium = "Premium"
    ideal = "Ideal"

# 2. On sécurise le schéma de données
class DiamondFeatures(BaseModel):
    # Field(gt=0) signifie "Greater Than 0" (strictement positif)
    carat: float = Field(..., gt=0, description="Poids du diamant en carats")
    cut: CutEnum  # Seules les valeurs de l'Enum sont acceptées
    color: str = Field(..., pattern="^[D-J]$") # Optionnel: restreint de D à J via Regex
    clarity: str
    depth: float = Field(..., gt=0)
    table: float = Field(..., gt=0)
    x: float = Field(..., gt=0)
    y: float = Field(..., gt=0)
    z: float = Field(..., gt=0)

@app.get("/")
def root():
    return {"status": "ok", "message": "Diamonds API is running 💎"}

@app.post("/predict")
def predict(features: DiamondFeatures):
    # model_dump(mode='json') assure que l'Enum est converti en string proprement
    X = pd.DataFrame([features.model_dump()])
    prediction = model.predict(X)
    return {"predicted_price": round(float(prediction[0]), 2)}