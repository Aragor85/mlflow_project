import mlflow.pyfunc
import tensorflow_hub as hub
import tensorflow as tf

# Chargement du modèle MLflow
def load_model():
    model_uri = "mlruns/763161070789444748/6297166d13dc418f8128dfd3e9cead67/artifacts/model"
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    
    # Chargement du module USE
    use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    return model, use_model

# Prédiction avec encodage USE
def predict_sentiment(model, use_model, text: str):
    vect_text = use_model([text])  # Encodage en vecteur
    prediction = model.predict(vect_text.numpy())  # Conversion en numpy si nécessaire
    return int(prediction[0])
