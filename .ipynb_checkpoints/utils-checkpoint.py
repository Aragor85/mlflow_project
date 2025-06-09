# utils.py
import pandas as pd
import numpy as np
import os
import re
import string
import yaml

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Chargement de la config ===
def load_config():
    config_path = "config.yml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# === Nettoyage des textes ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    return text.strip()

# === Données TF-IDF ===
def load_data_tfidf():
    path = r"C:\Users\Djamel\Desktop\Formation\Module_7\mlflow_project\data"
    df = pd.read_csv(os.path.join(path, 'sampled_sentiment140.csv'), encoding='latin-1')

    df['comment'] = df['comment'].astype(str).apply(clean_text)
    X = df['comment']
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer

# === Données USE ===
def load_data_use():
    path = r"C:\Users\Djamel\Desktop\Formation\Module_7\mlflow_project\data"
    df = pd.read_csv(os.path.join(path, 'sampled_sentiment140.csv'), encoding='latin-1')

    df['comment'] = df['comment'].astype(str).apply(clean_text)
    X = df['comment'].values
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🔄 Embedding avec Universal Sentence Encoder...")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    X_train_embed = embed(X_train).numpy()
    X_test_embed = embed(X_test).numpy()

    return X_train_embed, X_test_embed, y_train, y_test

# === Données LSTM ===
def load_data_lstm():
    path = r"C:\Users\Djamel\Desktop\Formation\Module_7\mlflow_project\data"
    df = pd.read_csv(os.path.join(path, 'sampled_sentiment140.csv'), encoding='latin-1')

    df['comment'] = df['comment'].astype(str).apply(clean_text)
    X = df['comment'].values
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    max_words = config["lstm"]["max_num_words"]
    max_len = config["lstm"]["max_sequence_length"]

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

    return X_train_pad, X_test_pad, y_train, y_test, tokenizer


# === Données BERT ===

def load_data_bert(tokenizer, max_len=128):
    """
    Charge les données, tokenize avec BERT, split train/test
    et retourne des dicts de tf.Tensor prêts pour model.fit().
    """
    # 1. Charger le CSV
    df = pd.read_csv(
        r"C:\Users\Djamel\Desktop\Formation\Module_7\mlflow_project\data\sampled_sentiment140.csv",
        encoding='latin-1'
    )
    # Nettoyage basique
    df['comment'] = df['comment'].astype(str).str.lower()
    
    texts = df['comment'].tolist()
    labels = df['target'].astype(int).tolist()

    # 2. Tokenization BERT
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors="tf"
    )
    input_ids = encodings['input_ids']           # shape (N, max_len)
    attention_mask = encodings['attention_mask'] # shape (N, max_len)
    labels_tensor = tf.constant(labels, dtype=tf.int32)  # shape (N,)

    # 3. Conversion numpy + split
    ids_np = input_ids.numpy()
    mask_np = attention_mask.numpy()
    labels_np = labels_tensor.numpy()

    ids_tr, ids_te, mask_tr, mask_te, y_tr, y_te = train_test_split(
        ids_np, mask_np, labels_np, test_size=0.2, random_state=42, stratify=labels_np
    )

    # 4. Reconversion en tf.Tensor
    X_train = {
        'input_ids':  tf.convert_to_tensor(ids_tr),
        'attention_mask': tf.convert_to_tensor(mask_tr)
    }
    X_test = {
        'input_ids':  tf.convert_to_tensor(ids_te),
        'attention_mask': tf.convert_to_tensor(mask_te)
    }
    y_train = tf.convert_to_tensor(y_tr, dtype=tf.int32)
    y_test  = tf.convert_to_tensor(y_te, dtype=tf.int32)

    return X_train, X_test, y_train, y_test