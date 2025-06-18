# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from app.model_utils import load_model, predict_sentiment

app = FastAPI()

# Chargement modèle + encodeur USE
model, use_model = load_model()

class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Bienvenue dans l'API d'analyse des sentiments (modèle USE)"}

@app.post("/predict")
def predict(input: TextInput):
    prediction = predict_sentiment(model, use_model, input.text)
    return {"prediction": prediction}


