from fastapi import FastAPI
from pydantic import BaseModel
from app.model_utils import load_model, predict_sentiment
from typing import Literal
import logging

# --- Configuration Application Insights ---
from opencensus.ext.azure.log_exporter import AzureLogHandler

# Créer un logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remplacer par ta vraie Instrumentation Key
logger.addHandler(
    AzureLogHandler(
        connection_string="InstrumentationKey=0fdd4361-7b38-476f-a2b5-eefc99fefbb0"
    )
)

# --- Initialisation FastAPI + chargement modèle ---
app = FastAPI()

# Charger modèle + Universal Sentence Encoder
model, use_model = load_model()

# --- Schémas Pydantic ---
class TextInput(BaseModel):
    Tweet: str

class Feedback(BaseModel):
    Tweet: str
    predicted_label: Literal["positif", "negatif"]
    correct_label: Literal["positif", "negatif"]

# --- Routes ---
@app.get("/")
def read_root():
    return {"message": "Bienvenue dans l'API d'analyse des sentiments (modèle USE)"}

@app.post("/predict")
def predict(input: TextInput):
    prediction = predict_sentiment(model, use_model, input.Tweet)

    label = "avis positif" if prediction == 1 else "avis négatif"

    return {
        "prediction": int(prediction),
        "label": label
    }

@app.post("/feedback")
def feedback(feedback: Feedback):
    if feedback.predicted_label != feedback.correct_label:
        logger.warning(
            "⚠️ Feedback utilisateur : mauvaise prédiction du modèle.",
            extra={
                "custom_dimensions": {
                    "input_text": feedback.Tweet,
                    "predicted_label": feedback.predicted_label,
                    "correct_label": feedback.correct_label,
                    "type": "model_misclassification"
                }
            }
        )
    else:
        logger.info(
            "✅ Feedback utilisateur : prédiction correcte confirmée.",
            extra={
                "custom_dimensions": {
                    "input_text": feedback.Tweet,
                    "predicted_label": feedback.predicted_label,
                    "correct_label": feedback.correct_label,
                    "type": "model_validation_success"
                }
            }
        )
    return {
        "message": "Merci pour votre retour. Votre signalement a été enregistré pour améliorer le modèle."
    }
