import mlflow.pyfunc
import tensorflow_hub as hub
import tensorflow as tf

# Chargement du modèle MLflow depuis le chemin local
def load_model():
    model_uri = "app/model"  # ✅ modèle local dans le répertoire app/model
    model = mlflow.pyfunc.load_model(model_uri=model_uri)

    # Chargement du module Universal Sentence Encoder depuis TensorFlow Hub
    use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    return model, use_model

# Prédiction avec encodage USE
def predict_sentiment(model, use_model, text: str):
    vect_text = use_model([text])  # ✅ encodage du texte
    prediction = model.predict(vect_text.numpy())  # ✅ conversion Tensor -> numpy
    return int(prediction[0])
