# bert_model.py

import tensorflow as tf
import mlflow
import mlflow.tensorflow
from transformers import TFBertForSequenceClassification, BertTokenizer
from utils import load_data_bert, load_config

def build_and_train_bert():
    config = load_config()
    bert_cfg = config["bert"]

    model_name = bert_cfg.get("model_name", "bert-base-uncased")
    max_len = bert_cfg.get("max_sequence_length")
    batch_size = bert_cfg.get("batch_size")
    epochs = bert_cfg.get("epochs")
    learning_rate = bert_cfg.get("learning_rate")

    # Initialiser tokenizer BERT
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Charger les données
    print("📥 Chargement des données BERT...")
    X_train, X_test, y_train, y_test = load_data_bert(tokenizer, max_len)

    # Initialiser le modèle BERT
    print("🧠 Initialisation du modèle BERT...")
    model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    # Entraînement avec MLflow
    with mlflow.start_run(run_name="BERT_Classifier"):
        mlflow.log_params({
            "model_name": model_name,
            "max_len": max_len,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate
        })

        print("🚀 Entraînement en cours...")
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=batch_size,
            epochs=epochs
        )

        print("✅ Évaluation...")
        loss, acc = model.evaluate(X_test, y_test)
        mlflow.log_metrics({"test_loss": loss, "test_accuracy": acc})

        # Sauvegarde du modèle
        model.save_pretrained("models/bert_model")
        tokenizer.save_pretrained("models/bert_model")
        mlflow.tensorflow.log_model(tf_saved_model_dir="models/bert_model", tf_meta_graph_tags=None, tf_signature_def_key=None, artifact_path="bert_model")

        print(f"🎯 Test Accuracy: {acc:.4f}")
