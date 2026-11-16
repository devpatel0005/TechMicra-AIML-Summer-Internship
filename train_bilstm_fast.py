"""
Fast Bi-LSTM training for Stock Sentiment Analysis.
Trains on a sample of the data so it completes quickly (CPU-friendly).
Run once to generate models/bilstm_model.keras and models/tokenizer.pkl.

Run:  python train_bilstm_fast.py
"""

import os
import pickle
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = "Datasets/Sentiment_Stock_data.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/devpatel0005/TechMicra-AIML-Summer-Internship/"
    "refs/heads/main/Datasets/Sentiment_Stock_data.csv"
)
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MAX_WORDS = 10000
MAX_LEN = 100
EMBEDDING_DIM = 128
SAMPLE_SIZE = 30000   # train on a sample for speed (stratified)
EPOCHS = 3            # fewer epochs for speed
BATCH_SIZE = 64


def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_csv(DATA_URL)

    df = df.dropna().drop_duplicates()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["Sentence"] = df["Sentence"].astype(str)
    return df.reset_index(drop=True)


def main():
    print("Loading dataset...")
    df = load_data()
    print(f"Full dataset shape: {df.shape}")

    # Stratified sample for fast training
    df_sample, _ = train_test_split(
        df, train_size=SAMPLE_SIZE, random_state=42, stratify=df["Sentiment"]
    )
    print(f"Training on stratified sample: {len(df_sample):,} rows")

    # Tokenize
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df_sample["Sentence"])
    seq = tokenizer.texts_to_sequences(df_sample["Sentence"])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    y = df_sample["Sentiment"].values

    X_train, X_test, y_train, y_test = train_test_split(
        padded, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")

    # Build model (same architecture as notebook)
    model = Sequential(
        [
            Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.5),
            Bidirectional(LSTM(32)),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("\nTraining Bi-LSTM (fast mode)...")
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1,
    )

    # Evaluate
    probs = model.predict(X_test, verbose=0)
    y_pred = (probs > 0.5).astype(int).ravel()
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred, target_names=["Negative", "Positive"], output_dict=True
    )
    print(f"\nBi-LSTM (fast) Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    # Save metrics (pre-computed, for UI to display without live testing)
    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report,
    }
    with open(os.path.join(MODEL_DIR, "lstm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save artifacts
    model.save(os.path.join(MODEL_DIR, "bilstm_model.keras"))
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"\nSaved: {MODEL_DIR}/bilstm_model.keras")
    print(f"Saved: {MODEL_DIR}/tokenizer.pkl")
    print("\nDone! You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
