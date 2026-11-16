# 📈 Stock Sentiment Analyzer — TechMicra AI/ML Summer Internship

> **A complete end-to-end Stock Sentiment Analysis project with a production-ready Streamlit UI, two trained machine learning models (Random Forest + Bi-LSTM), pre-computed evaluation metrics, and zero live training at deployment time.**

---

## 🌐 Live Demo

**[🚀 Try the Deployed App Here](https://techmicra-aiml-summer-internship-xsrsjsstscul6htdccvwfu.streamlit.app/)**

## ✍️ Project Blog

**[📖 Read the Full Blog on Medium](https://medium.com/@23aiml047/from-financial-news-to-market-sentiment-building-an-end-to-end-stock-sentiment-analyzer-during-my-4e9a56c76cc6)**

> A detailed walkthrough of building this end-to-end Stock Sentiment Analyzer — from financial news data to market sentiment predictions.

> Built by **Dev Patel** | Student ID: **23AIML047** | TechMicra AI/ML Summer Internship

---

## 📑 Table of Contents

1. [Abstract](#abstract)
2. [Problem Statement](#problem-statement)
3. [Project Objectives](#project-objectives)
4. [Scope of the Project](#scope-of-the-project)
5. [Methodology](#methodology)
6. [Dataset Description](#dataset-description)
7. [Data Preprocessing](#data-preprocessing)
8. [Feature Engineering](#feature-engineering)
9. [Models Implemented](#models-implemented)
   - [9.1 Random Forest (TF-IDF + Numeric Features)](#91-random-forest)
   - [9.2 Bi-LSTM (Deep Learning)](#92-bi-lstm)
10. [Hyperparameter Tuning](#hyperparameter-tuning)
11. [Model Evaluation & Test Metrics](#model-evaluation--test-metrics)
12. [Streamlit UI Features](#streamlit-ui-features)
13. [Project File Structure](#project-file-structure)
14. [Installation & Setup](#installation--setup)
15. [How to Run Locally](#how-to-run-locally)
16. [Offline Training Scripts](#offline-training-scripts)
17. [Deployment on Streamlit Cloud](#deployment-on-streamlit-cloud)
18. [Results & Findings](#results--findings)
19. [Limitations](#limitations)
20. [Future Scope](#future-scope)
21. [Author & Attribution](#author--attribution)
22. [License](#license)
23. [References](#references)

---

## 📄 Abstract

Stock market movements are heavily influenced by public sentiment expressed in financial news. This project, **Stock Sentiment Analyzer**, presents a machine learning-based solution that classifies financial news sentences into **Positive** or **Negative** sentiment categories. The system is built as an end-to-end pipeline covering Exploratory Data Analysis (EDA), text preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, and finally a user-friendly **Streamlit web interface**.

Two models were developed and compared:
1. **Random Forest Classifier** using TF-IDF vectorized text combined with handcrafted numeric features (sentence length and word count).
2. **Bi-Directional LSTM (Bi-LSTM)** deep learning model using tokenized text sequences and embedding layers.

To ensure instant deployment without any runtime training overhead, both models are **pre-trained offline** and their artifacts (model files, vectorizers, tokenizers, and pre-computed test metrics) are stored in the repository. The Streamlit UI loads these saved artifacts directly — **no live training or live testing occurs during deployment**, guaranteeing fast startup and zero errors.

The deployed application is live at:  
👉 https://techmicra-aiml-summer-internship-xsrsjsstscul6htdccvwfu.streamlit.app/

---

## 🎯 Problem Statement

Financial traders, investors, and analysts need to quickly gauge market sentiment from news headlines and articles. Manual sentiment analysis is time-consuming and subjective. There is a need for an automated system that can:

- Accept a financial news sentence as input.
- Predict whether the sentiment is Positive or Negative.
- Provide a confidence score for the prediction.
- Offer a simple, interactive web interface for non-technical users.
- Be deployed online with no dependency on live model training (which is computationally expensive).

---

## 🎯 Project Objectives

1. Load and explore a financial sentiment dataset (108,750 labeled sentences).
2. Perform text preprocessing including null handling, duplicate removal, and text cleaning.
3. Engineer numerical features (sentence length, word count) alongside text features.
4. Convert text into numerical representations using TF-IDF (Random Forest) and Tokenizer + Embedding (Bi-LSTM).
5. Train a Random Forest classifier with hyperparameter tuning via `RandomizedSearchCV`.
6. Train a Bi-LSTM deep learning model with Embedding and Bidirectional LSTM layers.
7. Evaluate both models using test accuracy, confusion matrix, and classification report.
8. Save all trained models and pre-computed metrics to disk for reuse.
9. Build a Streamlit web application that loads pre-trained models — **no live training**.
10. Deploy the application on Streamlit Cloud for public access.

---

## 📊 Dataset Description

| Attribute | Details |
|-----------|---------|
| **File Name** | `Sentiment_Stock_data.csv` |
| **Location** | `Datasets/Sentiment_Stock_data.csv` |
| **Total Rows** | 108,750 sentences |
| **Columns** | `Sentiment`, `Sentence`, `Sentence_Length`, `Word_Count` |
| **Sentiment Labels** | `0` = Negative, `1` = Positive |
| **Positive Count** | ~55,724 |
| **Negative Count** | ~53,026 |
| **Class Balance** | Nearly balanced (slight class imbalance) |
| **Source** | GitHub raw URL (fallback if local file missing) |

The dataset contains financial news sentences labeled with binary sentiment. Each sentence is accompanied by its character length and word count, which are used as auxiliary numeric features for the Random Forest model.

---

## 🔧 Data Preprocessing

The following preprocessing steps were applied (as implemented in `app.py` and the training scripts):

1. **Load dataset** from local CSV or GitHub raw URL fallback.
2. **Drop missing values** — rows with null `Sentence` were removed (`dropna()`).
3. **Remove duplicates** — exact duplicate rows were eliminated (`drop_duplicates()`).
4. **Remove irrelevant columns** — the `Unnamed: 0` index column was dropped if present.
5. **Feature creation** — two numeric columns were generated:
   - `Sentence_Length` = `len(sentence)` (character count)
   - `Word_Count` = `len(sentence.split())` (word count)
6. **Type casting** — the `Sentence` column was cast to string type to prevent tokenization errors on non-string entries.
7. **Train-test split** — stratified split with `test_size=0.2`, `random_state=42` to maintain class distribution.

After preprocessing, the final dataset contained **108,750 rows × 4 columns**.

---

## ⚙️ Feature Engineering

### For Random Forest Model
The model uses a **hybrid feature vector** combining:

| Feature Type | Source | Dimensions |
|--------------|--------|------------|
| TF-IDF Text Features | `TfidfVectorizer(max_features=5000)` | 5,000 |
| Numeric Feature 1 | Sentence Length (characters) | 1 |
| Numeric Feature 2 | Word Count | 1 |
| **Total** | | **5,002** |

TF-IDF (Term Frequency-Inverse Document Frequency) converts each sentence into a 5,000-dimensional sparse vector representing the most important words in the corpus. The two numeric features are appended to create a combined feature matrix of shape `(n_samples, 5002)`.

### For Bi-LSTM Model
The model uses **tokenized word sequences**:

| Parameter | Value |
|-----------|-------|
| `max_words` (vocabulary size) | 10,000 |
| `max_len` (sequence length) | 100 |
| `oov_token` | `"<OOV>"` |
| Padding | `post` |
| Truncating | `post` |
| Embedding dimension | 128 |

Each sentence is converted to a sequence of integers (word indices), padded/truncated to length 100, and fed into an Embedding layer that maps words to 128-dimensional dense vectors.

---

## 🤖 Models Implemented

### 9.1 Random Forest (TF-IDF + Numeric Features)

**Architecture:**
```
Input: Sentence text
├── TF-IDF Vectorizer (max_features=5000) → 5,000-dim sparse vector
├── Sentence Length → 1 numeric feature
├── Word Count → 1 numeric feature
└── Combined Feature Vector → 5,002 dimensions
       ↓
Random Forest Classifier (n_estimators=100, max_depth=10, ...)
       ↓
Output: Binary Sentiment (0=Negative, 1=Positive)
```

**Hyperparameters (from RandomizedSearchCV best run):**
```python
{
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}
```

**Saved artifacts:**
- `models/rf_model.pkl`
- `models/tfidf_vectorizer.pkl`
- `models/rf_metrics.json`

---

### 9.2 Bi-LSTM (Deep Learning)

**Architecture:**
```
Input: Tokenized & Padded Sequence (100,)
       ↓
Embedding Layer (input_dim=10000, output_dim=128, input_length=100)
       ↓
Bidirectional LSTM (64 units, return_sequences=True)
       ↓
Dropout (0.5)
       ↓
Bidirectional LSTM (32 units)
       ↓
Dropout (0.5)
       ↓
Dense Layer (1 unit, sigmoid activation)
       ↓
Output: Probability (0 to 1) → threshold 0.5 → Binary Sentiment
```

**Model Parameters:**
- Total Parameters: ~4.26 million
- Trainable Parameters: ~1.42 million
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Metric: Accuracy
- Epochs: 5
- Batch Size: 32
- Validation Split: 0.1

**Saved artifacts:**
- `models/bilstm_model.keras`
- `models/tokenizer.pkl`
- `models/lstm_metrics.json`

---

## 🔧 Hyperparameter Tuning

The Random Forest model was optimized using `RandomizedSearchCV` with the following search space:

```python
param_grid = {
    'n_estimators': [10, 30, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
```

Configuration:
- `n_iter=3` (3 random combinations)
- `cv=2` (2-fold cross-validation)
- `scoring='f1'`
- `n_jobs=-1` (use all CPU cores)
- `random_state=42`

**Best parameters found:**
```python
{'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_depth': 10}
```

---

## 📈 Model Evaluation & Test Metrics

**Important:** The Streamlit UI does **not** perform live testing. All metrics below are **pre-computed during offline training** and loaded from JSON files (`rf_metrics.json`, `lstm_metrics.json`) at runtime.

### Random Forest Test Metrics
| Metric | Value |
|--------|-------|
| Accuracy | ~53.77% |
| Negative Precision | ~0.65 |
| Negative Recall | ~0.11 |
| Negative F1-Score | ~0.19 |
| Positive Precision | ~0.53 |
| Positive Recall | ~0.94 |
| Positive F1-Score | ~0.68 |

### Bi-LSTM Test Metrics
| Metric | Value |
|--------|-------|
| Accuracy | ~51.75% |
| Negative Precision | ~0.52 |
| Negative Recall | ~0.52 |
| Negative F1-Score | ~0.51 |
| Positive Precision | ~0.54 |
| Positive Recall | ~0.56 |
| Positive F1-Score | ~0.55 |

> **Note:** Both models show modest accuracy. The Random Forest tends to over-predict the Positive class (high recall, low precision for Negative). The Bi-LSTM is more balanced but still struggles with class separation. These results are suitable for demonstration and learning purposes — **not for actual financial trading decisions**.

### Confusion Matrices
Both models display confusion matrices in the UI:
- **Random Forest:** Green heatmap
- **Bi-LSTM:** Purple heatmap

Axes: Actual (rows) vs Predicted (columns) with labels "Negative" and "Positive".

---

## 🖥️ Streamlit UI Features

| Feature | Description |
|---------|-------------|
| **Model Selection** | Sidebar radio button: Random Forest vs Bi-LSTM |
| **Sentiment Prediction** | Text input → Analyze Sentiment → Positive/Negative + confidence score |
| **Confidence Score** | Probability percentage with progress bar |
| **Sample News Dropdown** | 6 pre-written realistic financial headlines for instant testing |
| **Clear Button** | Resets the text input field |
| **Model Evaluation Metrics** | Toggleable sidebar checkbox: shows test accuracy, classification report, confusion matrix (pre-computed) |
| **Dataset EDA** | Toggleable sidebar checkbox: sentiment distribution, sentence length histogram, word count histogram, sample data table |
| **Attribution** | "Built by Dev Patel — 23AIML047" displayed at top left |
| **No Live Training** | Models load from disk; app starts instantly |

### Sample Headlines Included in Dropdown
1. Apple reports record quarterly revenue as iPhone sales exceed expectations
2. Tesla stock plunges after CEO announces production delays and missed delivery targets
3. Federal Reserve signals potential interest rate cut amid cooling inflation data
4. Amazon shares surge following strong AWS cloud computing growth announcement
5. Regional bank shares tumble on reports of significant loan defaults and credit losses
6. Pharmaceutical stock soars after successful phase 3 clinical trial results for new drug

---

## 📁 Project File Structure

```
TechMicra-AIML-Summer-Internship/
│
├── 📄 app.py                              # Streamlit UI application (main entry point)
├── 📄 requirements.txt                    # Python dependencies for deployment
├── 📄 README.md                           # Project documentation (this file)
├── 📄 .gitignore                          # Git ignore rules
│
├── 📓 Stock_Sentiment_Analysis_and_Prediction.ipynb   # Original Colab notebook (EDA + training)
│
├── 📂 Datasets/
│   └── 📄 Sentiment_Stock_data.csv        # Training dataset (108,750 rows)
│
├── 📂 models/                             # Pre-trained model artifacts (NO live training in UI)
│   ├── 📄 rf_model.pkl                    # Random Forest classifier (joblib)
│   ├── 📄 tfidf_vectorizer.pkl            # TF-IDF vectorizer for Random Forest
│   ├── 📄 rf_metrics.json                 # Pre-computed RF test metrics (accuracy, CM, report)
│   ├── 📄 bilstm_model.keras              # Bi-LSTM deep learning model (Keras format)
│   ├── 📄 tokenizer.pkl                   # Bi-LSTM tokenizer (pickle)
│   └── 📄 lstm_metrics.json               # Pre-computed Bi-LSTM test metrics
│
├── 🐍 train_and_save_models.py            # Full offline training script (RF + Bi-LSTM)
├── 🐍 train_bilstm_fast.py                # Fast offline training script (Bi-LSTM on 30k sample)
│
└── 📂 .venv/                              # Local virtual environment (ignored by git)
```

---

## 💾 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- `pip` package manager
- `venv` module (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/devpatel0005/TechMicra-AIML-Summer-Internship.git
cd TechMicra-AIML-Summer-Internship
```

### Step 2: Create Virtual Environment
```bash
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies:**
- `streamlit` — Web application framework
- `pandas` — Data manipulation
- `numpy` — Numerical computing
- `scikit-learn` — Machine learning (Random Forest, metrics, train_test_split)
- `scipy` — Sparse matrices (hstack)
- `matplotlib` — Plotting
- `seaborn` — Statistical visualization
- `tensorflow` — Deep learning (Bi-LSTM)
- `joblib` — Model serialization

---

## ▶️ How to Run Locally

```bash
# Activate virtual environment
source .venv/bin/activate    # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Run the Streamlit app
streamlit run app.py
```

The app will start and display a URL (typically `http://localhost:8507`). Open that URL in your browser.

**To stop the server:** Press `Ctrl+C` in the terminal.

---

## 🐍 Offline Training Scripts

If you want to re-train the models from scratch:

### Option A: Full Training (Random Forest + Bi-LSTM)
```bash
python train_and_save_models.py
```
- Trains Random Forest on full dataset (fast)
- Trains Bi-LSTM on full dataset with 5 epochs (slow, ~25+ minutes on CPU)
- Saves all model artifacts and metrics to `models/`

### Option B: Fast Bi-LSTM Training (Recommended for Quick Testing)
```bash
python train_bilstm_fast.py
```
- Trains Bi-LSTM on a stratified sample of 30,000 rows (3 epochs)
- Completes in ~2–3 minutes on CPU
- Saves Bi-LSTM model, tokenizer, and metrics to `models/`

> **Note:** You only need to run these scripts once. The Streamlit UI loads pre-trained models from `models/` and does **not** train at runtime.

---

## ☁️ Deployment on Streamlit Cloud

This app is fully prepared for deployment on [Streamlit Cloud](https://streamlit.io/cloud).

### Deployment Steps

1. **Push to GitHub** — Ensure this repository is on GitHub (main branch).
2. **Visit Streamlit Cloud** — Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **Create New App** — Click "New app" and configure:
   - **Repository:** `devpatel0005/TechMicra-AIML-Summer-Internship`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. **Deploy** — Click the Deploy button.

### Why No Errors During Deployment?
- ✅ All model files are pre-saved in `models/` (no runtime training).
- ✅ All dependencies are listed in `requirements.txt`.
- ✅ Dataset loads from local file or GitHub URL fallback.
- ✅ Metrics are loaded from JSON (no live testing).
- ✅ `.gitignore` excludes unnecessary files (`.venv`, `__pycache__`, etc.).

**Deployed App URL:**  
👉 https://techmicra-aiml-summer-internship-xsrsjsstscul6htdccvwfu.streamlit.app/

---

## 📊 Results & Findings

| Model | Test Accuracy | Best For |
|-------|--------------|----------|
| Random Forest (TF-IDF + Numeric) | ~53.77% | Fast training, interpretable features |
| Bi-LSTM (Deep Learning) | ~51.75% | Captures sequential text patterns |

**Key Observations:**
1. The Random Forest model achieves slightly higher accuracy than Bi-LSTM on this dataset.
2. Random Forest shows high recall for the Positive class (~94%) but low precision for Negative (~11%), meaning it over-predicts Positive.
3. Bi-LSTM is more balanced between classes but has lower overall accuracy.
4. Both models are suitable for educational demonstration but not for production financial decision-making.
5. Pre-computing metrics and saving model artifacts enables instant, error-free deployment.

---

## ⚠️ Limitations

1. **Modest Accuracy:** Both models achieve only ~52–54% test accuracy, which is close to random guessing for a binary task. More data, better features, or advanced models (BERT, RoBERTa) could improve results.
2. **Class Imbalance:** The dataset is slightly imbalanced, affecting precision/recall trade-offs.
3. **No Real-Time Data:** The model analyzes static text input, not live stock market data or real-time news feeds.
4. **Not Financial Advice:** Predictions are for educational/demonstration purposes only and should not be used for actual trading or investment decisions.
5. **English-Only:** The model is trained on English financial news only.

---

## 🔮 Future Scope

- Replace Bi-LSTM with transformer-based models (BERT, FinBERT, RoBERTa) for higher accuracy.
- Add real-time news scraping from financial websites (Moneycontrol, Bloomberg, Yahoo Finance).
- Integrate live stock price data and correlate sentiment with price movements.
- Add user authentication and history of past predictions.
- Deploy on AWS/GCP/Azure with Docker containerization.
- Add multi-language support for financial news.
- Implement ensemble methods combining Random Forest and Bi-LSTM predictions.

---

## 👨‍💻 Author & Attribution

**Built by:** Dev Patel  
**Student ID:** 23AIML047  
**Internship:** TechMicra AI/ML Summer Internship  
**Project:** Stock Sentiment Analysis with Streamlit UI

**Deployed App:** https://techmicra-aiml-summer-internship-xsrsjsstscul6htdccvwfu.streamlit.app/  
**GitHub Repository:** https://github.com/devpatel0005/TechMicra-AIML-Summer-Internship

---

## 📄 License

This project is developed as part of the TechMicra AI/ML Summer Internship. All rights reserved by the author.

---

## 📚 References

1. Scikit-learn Documentation — https://scikit-learn.org/
2. TensorFlow/Keras Documentation — https://www.tensorflow.org/api_docs
3. Streamlit Documentation — https://docs.streamlit.io/
4. TfidfVectorizer — scikit-learn feature extraction
5. RandomizedSearchCV — scikit-learn model selection
6. Bidirectional LSTM — Keras layers API
7. Original Notebook: `Stock_Sentiment_Analysis_and_Prediction.ipynb` (included in repository)

---

> **Built with ❤️ for the TechMicra AI/ML Summer Internship by Dev Patel (23AIML047)**
