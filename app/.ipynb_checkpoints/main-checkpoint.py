# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.model_utils import load_model, predict_sentiment
from typing import Literal
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler

# --- Logger Application Insights ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(
    AzureLogHandler(
        connection_string="InstrumentationKey=0fdd4361-7b38-476f-a2b5-eefc99fefbb0"
    )
)

# --- Application FastAPI ---
app = FastAPI()

# Charger modèle + Universal Sentence Encoder
model, use_model = load_model()

# --- Schémas d’entrée ---
class TextInput(BaseModel):
    text: str = Field(..., alias="tweet", description="Please insert your tweet about Air Paradis")

class Feedback(BaseModel):
    text: str
    predicted_label: Literal["positif", "negatif"]
    correct_label: Literal["positif", "negatif"]

# --- Routes ---
@app.get("/")
def read_root():
    return {"message": "Bienvenue dans l'API d'analyse des sentiments (modèle USE)"}

@app.post("/predict")
def predict(input: TextInput):
    prediction = predict_sentiment(model, use_model, input.text)
    return {"prediction": prediction}

@app.post("/feedback")
def feedback(feedback: Feedback):
    logger.warning(
        "⚠️ Feedback utilisateur : mauvaise prédiction du modèle.",
        extra={
            "custom_dimensions": {
                "input_text": feedback.text,
                "predicted_label": feedback.predicted_label,
                "correct_label": feedback.correct_label,
                "type": "model_misclassification"
            }
        }
    )
    return {
        "message": "Merci pour votre retour. Votre signalement a été enregistré pour améliorer le modèle."
    }
