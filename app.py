"""
Streamlit UI for Stock Sentiment Analysis
=========================================
Models reproduced from Stock_Sentiment_Analysis_and_Prediction.ipynb:

1. Random Forest  -> TF-IDF (max_features=5000) + numeric features
                    (Sentence_Length, Word_Count), hyperparameters from the
                    notebook's RandomizedSearchCV best run.
2. Bi-LSTM        -> Tokenizer (max_words=10000, max_len=100),
                    Embedding(128) + 2x Bidirectional LSTM + sigmoid.

Run:  streamlit run app.py
"""

import os
import pickle
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = "Datasets/Sentiment_Stock_data.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/devpatel0005/TechMicra-AIML-Summer-Internship/"
    "refs/heads/main/Datasets/Sentiment_Stock_data.csv"
)

MODEL_DIR = "models"

# Best hyperparameters found by the notebook's RandomizedSearchCV.
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

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_csv(DATA_URL)

    df = df.dropna().drop_duplicates()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Recreate numeric features used by the Random Forest model.
    if "Sentence_Length" not in df.columns:
        df["Sentence_Length"] = df["Sentence"].astype(str).apply(len)
    if "Word_Count" not in df.columns:
        df["Word_Count"] = df["Sentence"].astype(str).apply(
            lambda x: len(str(x).split())
        )

    df["Sentence"] = df["Sentence"].astype(str)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Random Forest pipeline (load pre-trained model from disk — no live training)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Random Forest model from disk...")
def load_rf():
    model_path = os.path.join(MODEL_DIR, "rf_model.pkl")
    vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    if not (os.path.exists(model_path) and os.path.exists(vec_path)):
        raise FileNotFoundError(
            f"Pre-trained Random Forest files not found in '{MODEL_DIR}/'. "
            "Run 'python train_and_save_models.py' first to generate them."
        )
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return {"model": model, "vectorizer": vectorizer}


def predict_rf(text, rf):
    vec = rf["vectorizer"].transform([str(text)])
    length = len(str(text))
    word_count = len(str(text).split())
    num = np.array([[length, word_count]])
    X = hstack([vec, num])
    prob = float(rf["model"].predict_proba(X)[0][1])
    pred = int(prob >= 0.5)
    return pred, prob


# ---------------------------------------------------------------------------
# Bi-LSTM pipeline (load pre-trained model from disk — no live training)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Bi-LSTM model from disk...")
def load_lstm():
    model_path = os.path.join(MODEL_DIR, "bilstm_model.keras")
    tok_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
    if not (os.path.exists(model_path) and os.path.exists(tok_path)):
        raise FileNotFoundError(
            f"Pre-trained Bi-LSTM files not found in '{MODEL_DIR}/'. "
            "Run 'python train_and_save_models.py' first to generate them."
        )
    model = tf.keras.models.load_model(model_path)
    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)
    return {"model": model, "tokenizer": tokenizer}


def predict_lstm(text, lstm):
    seq = lstm["tokenizer"].texts_to_sequences([str(text)])
    padded = pad_sequences(
        seq, maxlen=MAX_LEN, padding="post", truncating="post"
    )
    prob = float(lstm["model"].predict(padded, verbose=0)[0][0])
    pred = int(prob >= 0.5)
    return pred, prob


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📈 Stock Sentiment Analyzer")
st.markdown(
    "Predict whether a financial news sentence expresses a **Positive** or "
    "**Negative** sentiment. Pick a model, type a sentence, and get an instant "
    "prediction with confidence."
)

# Built by attribution (top left, small font)
st.markdown(
    "<div style='font-size: 0.85rem; color: #666; margin-top: -0.5rem;'>"
    "<strong>Built by Dev Patel</strong><br>"
    "23AIML047"
    "</div>",
    unsafe_allow_html=True,
)

# Load dataset
with st.spinner("Loading dataset..."):
    df = load_data()

st.success(f"Dataset loaded: {df.shape[0]:,} sentences × {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.radio(
    "Select model",
    [
        "Random Forest (TF-IDF + numeric features)",
        "Bi-LSTM (deep learning)",
    ],
    index=0,
)
show_metrics = st.sidebar.checkbox("Show model evaluation metrics", value=False)
show_eda = st.sidebar.checkbox("Show dataset EDA", value=True)

use_lstm = model_choice.startswith("Bi-LSTM")

st.sidebar.info(
    "Models reproduce the notebook's results, which show modest accuracy "
    "(Random Forest ~54%, Bi-LSTM ~53% on the test set). Treat predictions "
    "as a demonstration, not financial advice."
)

# Load pre-trained model from disk (no live training)
if use_lstm:
    try:
        lstm = load_lstm()
        st.session_state["lstm"] = lstm
        st.sidebar.success("Bi-LSTM loaded from disk ✅")
    except FileNotFoundError as e:
        lstm = None
        st.sidebar.error("Bi-LSTM model not found")
        st.sidebar.caption(str(e))
else:
    try:
        rf = load_rf()
        st.session_state["rf"] = rf
        st.sidebar.success("Random Forest loaded from disk ✅")
    except FileNotFoundError as e:
        rf = None
        st.sidebar.error("Random Forest model not found")
        st.sidebar.caption(str(e))

# ---------------------------------------------------------------------------
# Sample news dropdown
# ---------------------------------------------------------------------------
SAMPLE_HEADLINES = [
    "Apple reports record quarterly revenue as iPhone sales exceed expectations",
    "Tesla stock plunges after CEO announces production delays and missed delivery targets",
    "Federal Reserve signals potential interest rate cut amid cooling inflation data",
    "Amazon shares surge following strong AWS cloud computing growth announcement",
    "Regional bank shares tumble on reports of significant loan defaults and credit losses",
    "Pharmaceutical stock soars after successful phase 3 clinical trial results for new drug",
]

with st.expander("📰 Quick-select sample news (optional)"):
    sample_choice = st.selectbox(
        "Choose a sample headline:",
        ["-- Select a headline --"] + SAMPLE_HEADLINES,
        index=0,
    )
    if sample_choice != "-- Select a headline --":
        st.session_state["sentiment_text"] = sample_choice

# ---------------------------------------------------------------------------
# Prediction section
# ---------------------------------------------------------------------------
st.subheader("🔮 Sentiment Prediction")

if "sentiment_text" not in st.session_state:
    st.session_state["sentiment_text"] = ""

text_input = st.text_area(
    "Enter a financial news sentence:",
    value=st.session_state["sentiment_text"],
    placeholder='e.g. "Apple reports record quarterly revenue as iPhone sales exceed expectations"',
    height=130,
    key="sentiment_text",
)

col1, col2 = st.columns([1, 1])
predict_clicked = col1.button(
    "Analyze Sentiment", type="primary", use_container_width=True
)
clear_clicked = col2.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state["sentiment_text"] = ""
    st.rerun()

if predict_clicked:
    if not text_input.strip():
        st.warning("Please enter a sentence first.")
    else:
        if use_lstm:
            if lstm is None:
                st.error(
                    "Bi-LSTM model not found. Run 'python train_and_save_models.py' first."
                )
            else:
                pred, prob = predict_lstm(text_input, lstm)
        else:
            if rf is None:
                st.error(
                    "Random Forest model not found. Run 'python train_and_save_models.py' first."
                )
            else:
                pred, prob = predict_rf(text_input, rf)

        label = "Positive 🟢" if pred == 1 else "Negative 🔴"
        confidence = prob if pred == 1 else 1 - prob

        st.markdown(f"### Prediction: **{label}**")
        st.metric("Confidence", f"{confidence:.2%}")
        st.progress(confidence)
        st.caption(
            f"Model: {model_choice}  |  Raw positive-score: {prob:.4f}  "
            f"(threshold 0.50)"
        )

# ---------------------------------------------------------------------------
# Metrics section
# ---------------------------------------------------------------------------
if show_metrics:
    st.subheader("📊 Model Evaluation (Test Set)")
    if use_lstm:
        if lstm is None:
            st.info("Bi-LSTM model not found. Run 'python train_bilstm_fast.py' first.")
        else:
            # Load pre-computed metrics from offline training (no live testing)
            metrics_path = os.path.join(MODEL_DIR, "lstm_metrics.json")
            if not os.path.exists(metrics_path):
                st.warning(
                    "Pre-computed Bi-LSTM metrics not found. Run "
                    "'python train_bilstm_fast.py' to generate them."
                )
            else:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)

                acc = metrics["accuracy"]
                cm = np.array(metrics["confusion_matrix"])
                report = metrics["classification_report"]

                st.metric("Test Accuracy", f"{acc:.2%}")
                st.text(
                    f"precision    recall  f1-score   support\n\n"
                    f"Negative     {report['Negative']['precision']:.2f}      "
                    f"{report['Negative']['recall']:.2f}      "
                    f"{report['Negative']['f1-score']:.2f}      "
                    f"{int(report['Negative']['support'])}\n"
                    f"Positive     {report['Positive']['precision']:.2f}      "
                    f"{report['Positive']['recall']:.2f}      "
                    f"{report['Positive']['f1-score']:.2f}      "
                    f"{int(report['Positive']['support'])}\n\n"
                    f"accuracy                           {report['accuracy']:.2f}      "
                    f"{int(report['Negative']['support'] + report['Positive']['support'])}\n"
                    f"macro avg      {report['macro avg']['precision']:.2f}      "
                    f"{report['macro avg']['recall']:.2f}      "
                    f"{report['macro avg']['f1-score']:.2f}      "
                    f"{int(report['Negative']['support'] + report['Positive']['support'])}\n"
                    f"weighted avg   {report['weighted avg']['precision']:.2f}      "
                    f"{report['weighted avg']['recall']:.2f}      "
                    f"{report['weighted avg']['f1-score']:.2f}      "
                    f"{int(report['Negative']['support'] + report['Positive']['support'])}"
                )
                fig, ax = plt.subplots()
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Purples",
                    ax=ax,
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"],
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix (Bi-LSTM)")
                st.pyplot(fig)
    else:
        if rf is None:
            st.info(
                "Random Forest model not found. Run "
                "'python train_and_save_models.py' first."
            )
        else:
            # Load pre-computed metrics from offline training (no live testing)
            metrics_path = os.path.join(MODEL_DIR, "rf_metrics.json")
            if not os.path.exists(metrics_path):
                st.warning(
                    "Pre-computed Random Forest metrics not found. Run "
                    "'python train_and_save_models.py' to generate them."
                )
            else:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)

                acc = metrics["accuracy"]
                cm = np.array(metrics["confusion_matrix"])
                report = metrics["classification_report"]

                st.metric("Test Accuracy", f"{acc:.2%}")
                st.text(
                    f"precision    recall  f1-score   support\n\n"
                    f"Negative     {report['Negative']['precision']:.2f}      "
                    f"{report['Negative']['recall']:.2f}      "
                    f"{report['Negative']['f1-score']:.2f}      "
                    f"{int(report['Negative']['support'])}\n"
                    f"Positive     {report['Positive']['precision']:.2f}      "
                    f"{report['Positive']['recall']:.2f}      "
                    f"{report['Positive']['f1-score']:.2f}      "
                    f"{int(report['Positive']['support'])}\n\n"
                    f"accuracy                           {report['accuracy']:.2f}      "
                    f"{int(report['Negative']['support'] + report['Positive']['support'])}\n"
                    f"macro avg      {report['macro avg']['precision']:.2f}      "
                    f"{report['macro avg']['recall']:.2f}      "
                    f"{report['macro avg']['f1-score']:.2f}      "
                    f"{int(report['Negative']['support'] + report['Positive']['support'])}\n"
                    f"weighted avg   {report['weighted avg']['precision']:.2f}      "
                    f"{report['weighted avg']['recall']:.2f}      "
                    f"{report['weighted avg']['f1-score']:.2f}      "
                    f"{int(report['Negative']['support'] + report['Positive']['support'])}"
                )
                fig, ax = plt.subplots()
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Greens",
                    ax=ax,
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"],
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix (Random Forest)")
                st.pyplot(fig)

# ---------------------------------------------------------------------------
# EDA section
# ---------------------------------------------------------------------------
if show_eda:
    st.subheader("📈 Dataset Exploration")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total sentences", f"{len(df):,}")
    c2.metric("Positive (1)", f"{(df['Sentiment'] == 1).sum():,}")
    c3.metric("Negative (0)", f"{(df['Sentiment'] == 0).sum():,}")

    fig, ax = plt.subplots()
    sns.countplot(
        x="Sentiment",
        data=df,
        ax=ax,
        hue="Sentiment",
        legend=False,
        palette="viridis",
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Negative (0)", "Positive (1)"])
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["Sentence_Length"], bins=40, kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title("Sentence Length (characters)")
    axes[0].set_xlabel("Length")
    sns.histplot(df["Word_Count"], bins=40, kde=True, ax=axes[1], color="seagreen")
    axes[1].set_title("Word Count")
    axes[1].set_xlabel("Words")
    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("🔍 Sample data"):
        st.dataframe(df.sample(5, random_state=1)[["Sentiment", "Sentence"]])

st.divider()
st.caption(
    "Built for the TechMicra AI/ML Summer Internship · Stock Sentiment Analysis project"
)
