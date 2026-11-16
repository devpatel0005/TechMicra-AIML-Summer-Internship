"""
Offline training script for Stock Sentiment Analysis.
Runs once to pre-train and save both models so the Streamlit UI never trains live.

Run:  python train_and_save_models.py
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from scipy.sparse import hstack

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

RF_PARAMS = dict(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

MAX_WORDS = 10000
MAX_LEN = 100
EMBEDDING_DIM = 128


def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_csv(DATA_URL)

    df = df.dropna().drop_duplicates()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if "Sentence_Length" not in df.columns:
        df["Sentence_Length"] = df["Sentence"].astype(str).apply(len)
    if "Word_Count" not in df.columns:
        df["Word_Count"] = df["Sentence"].astype(str).apply(
            lambda x: len(str(x).split())
        )

    df["Sentence"] = df["Sentence"].astype(str)
    return df.reset_index(drop=True)


def train_and_save_rf(df):
    print("=" * 60)
    print("Training Random Forest (TF-IDF + numeric features)...")
    print("=" * 60)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf_vectorizer.fit_transform(df["Sentence"])
    numerical = df[["Sentence_Length", "Word_Count"]].values
    X = hstack([X_tfidf, numerical])
    y = df["Sentiment"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred, target_names=["Negative", "Positive"], output_dict=True
    )
    print(f"\nRandom Forest Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    # Save metrics (pre-computed, for UI to display without live testing)
    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report,
    }
    with open(os.path.join(MODEL_DIR, "rf_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save artifacts
    joblib.dump(model, os.path.join(MODEL_DIR, "rf_model.pkl"))
    joblib.dump(tfidf_vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    print(f"Saved: {MODEL_DIR}/rf_model.pkl")
    print(f"Saved: {MODEL_DIR}/tfidf_vectorizer.pkl")
    print(f"Saved: {MODEL_DIR}/rf_metrics.json")

    return model, tfidf_vectorizer, acc


def train_and_save_lstm(df):
    print("=" * 60)
    print("Training Bi-LSTM (this may take several minutes on CPU)...")
    print("=" * 60)

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["Sentence"])
    seq = tokenizer.texts_to_sequences(df["Sentence"])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    y = df["Sentiment"].values

    X_train, X_test, y_train, y_test = train_test_split(
        padded, y, test_size=0.2, random_state=42, stratify=y
    )

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
    model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
    )

    probs = model.predict(X_test, verbose=0)
    y_pred = (probs > 0.5).astype(int).ravel()
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred, target_names=["Negative", "Positive"], output_dict=True
    )
    print(f"\nBi-LSTM Test Accuracy: {acc:.4f}")
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
    print(f"Saved: {MODEL_DIR}/bilstm_model.keras")
    print(f"Saved: {MODEL_DIR}/tokenizer.pkl")

    return model, tokenizer, acc


def main():
    print("Loading dataset...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Positive (1): {(df['Sentiment'] == 1).sum():,}  "
          f"Negative (0): {(df['Sentiment'] == 0).sum():,}")

    train_and_save_rf(df)
    train_and_save_lstm(df)

    print("\n" + "=" * 60)
    print("All models trained and saved to the 'models/' folder.")
    print("You can now run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
