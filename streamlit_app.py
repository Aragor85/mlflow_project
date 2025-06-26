import streamlit as st
import requests

# URL de l'API FastAPI (utilise le nom DNS interne si Docker Compose, sinon localhost ou port 8000)
API_URL_PREDICT = "http://localhost:8000/predict"
API_URL_FEEDBACK = "http://localhost:8000/feedback"

st.title("💬 Analyse de sentiments")

# Initialiser les variables dans session_state
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "feedback_mode" not in st.session_state:
    st.session_state.feedback_mode = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "tweet_text" not in st.session_state:
    st.session_state.tweet_text = ""

# === Entrée du tweet ===
st.write("Entrez un tweet à analyser :")
tweet = st.text_area("📝 Tweet", value=st.session_state.tweet_text)

if st.button("Analyser"):
    if tweet.strip():
        try:
            response = requests.post(API_URL_PREDICT, json={"Tweet": tweet})
            if response.status_code == 200:
                result = response.json()
                pred = result["prediction"]
                label = "positif" if pred == 1 else "negatif"
                emoji = "👍" if pred == 1 else "👎"
                st.success(f"🔍 Prédiction : {emoji} **{label.upper()}**")

                # Stocker en session
                st.session_state.prediction_done = True
                st.session_state.prediction = label
                st.session_state.tweet_text = tweet
                st.session_state.feedback_mode = None  # Reset feedback

            else:
                st.error(f"❌ Erreur API : {response.status_code}")
        except Exception as e:
            st.error(f"❌ Erreur lors de la connexion à l'API : {e}")
    else:
        st.warning("⚠️ Veuillez entrer un tweet.")

# === Bloc feedback ===
if st.session_state.prediction_done:
    st.write("### 📣 Donnez votre feedback sur cette prédiction :")

    feedback = st.radio(
        "Cette prédiction vous semble-t-elle correcte ?",
        options=["Oui", "Non"],
        index=0 if st.session_state.feedback_mode != "Non" else 1
    )

    st.session_state.feedback_mode = feedback

    if feedback == "Oui":
        if st.button("✅ Confirmer que c’est correct"):
            try:
                requests.post(API_URL_FEEDBACK, json={
                    "Tweet": st.session_state.tweet_text,
                    "predicted_label": st.session_state.prediction,
                    "correct_label": st.session_state.prediction
                })
                st.success("Merci ! Votre validation a été enregistrée 🙌")
            except Exception as e:
                st.error(f"Erreur d'envoi du feedback : {e}")

    elif feedback == "Non":
        correct = st.radio("🔁 Quelle est la bonne prédiction ?", ["positif", "negatif"])
        if st.button("📩 Envoyer la correction"):
            try:
                requests.post(API_URL_FEEDBACK, json={
                    "Tweet": st.session_state.tweet_text,
                    "predicted_label": st.session_state.prediction,
                    "correct_label": correct
                })
                st.success("✅ Correction reçue, merci pour votre contribution.")
            except Exception as e:
                st.error(f"Erreur d'envoi du feedback : {e}")
