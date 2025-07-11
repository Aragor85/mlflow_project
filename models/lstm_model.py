# lstm_model.py

import mlflow
import mlflow.tensorflow
import tensorflow as tf
import yaml
import matplotlib.pyplot as plt
import os

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from utils import load_data_lstm, load_config


def plot_and_log_training_curves(history, output_path="lstm_training_curves.png"):
    """Génère et logue les courbes d'entraînement (accuracy et loss) dans MLflow"""
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Train vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # Log dans MLflow
    mlflow.log_artifact(output_path)


def train_lstm():
    config = load_config()
    lstm_cfg = config["lstm"]

    X_train, X_test, y_train, y_test, tokenizer = load_data_lstm()

    vocab_size = lstm_cfg["max_num_words"]
    max_len = lstm_cfg["max_sequence_length"]
    embedding_dim = lstm_cfg.get("embedding_dim")
    lstm_units = lstm_cfg.get("lstm_units")
    dropout_rate = lstm_cfg.get("dropout")
    batch_size = lstm_cfg.get("batch_size")
    epochs = lstm_cfg.get("epochs")

    with mlflow.start_run(run_name="Bidirectional_LSTM", nested=True):
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
        model.add(Bidirectional(LSTM(lstm_units)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=1
        )

        y_pred_probs = model.predict(X_test).flatten()
        y_pred = (y_pred_probs >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        loss = log_loss(y_test, y_pred_probs)
        roc_auc = roc_auc_score(y_test, y_pred_probs)

        mlflow.log_params({
            "vocab_size": vocab_size,
            "max_sequence_length": max_len,
            "embedding_dim": embedding_dim,
            "lstm_units": lstm_units,
            "dropout": dropout_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "bidirectional": True
        })

        mlflow.log_metrics({
            "accuracy": acc,
            "f1_score": f1,
            "log_loss": loss,
            "roc_auc": roc_auc
        })

        mlflow.keras.log_model(model, "Bidirectional_LSTM_Model")
        print(f"✅ Bidirectional LSTM terminé avec acc={acc:.2f} | f1={f1:.2f} | auc={roc_auc:.2f}")

        # 🔥 Générer et loguer les courbes
        plot_and_log_training_curves(history, output_path="lstm_training_curves.png")
