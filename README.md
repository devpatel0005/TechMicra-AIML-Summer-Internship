# TechMicra AI/ML Summer Internship — Stock Sentiment Analyzer

A complete Stock Sentiment Analysis project with a production-ready Streamlit UI. The project predicts whether a financial news sentence expresses **Positive** or **Negative** sentiment using two trained machine learning models.

## 🌐 Live Demo

**[🚀 Try the Deployed App Here](https://techmicra-aiml-summer-internship-xsrsjsstscul6htdccvwfu.streamlit.app/)**

> Built by **Dev Patel** | Student ID: **23AIML047**

---

## 📁 Project Structure

```
TechMicra-AIML-Summer-Internship/
│
├── app.py                              # Streamlit UI application
├── requirements.txt                    # Python dependencies for deployment
├── README.md                           # Project documentation (this file)
├── .gitignore                          # Git ignore rules
│
├── Stock_Sentiment_Analysis_and_Prediction.ipynb   # Original Colab notebook (EDA + training)
│
├── Datasets/
│   └── Sentiment_Stock_data.csv        # Training dataset (108,750 rows)
│
├── models/                             # Pre-trained model artifacts (no live training in UI)
│   ├── rf_model.pkl                    # Random Forest classifier
│   ├── tfidf_vectorizer.pkl            # TF-IDF vectorizer (Random Forest)
│   ├── rf_metrics.json                 # Pre-computed Random Forest test metrics
│   ├── bilstm_model.keras              # Bi-LSTM deep learning model
│   ├── tokenizer.pkl                   # Bi-LSTM tokenizer
│   └── lstm_metrics.json               # Pre-computed Bi-LSTM test metrics
│
├── train_and_save_models.py            # Offline training script (Random Forest + Bi-LSTM)
├── train_bilstm_fast.py                # Fast offline training script (Bi-LSTM on sample)
│
└── .venv/                              # Local virtual environment (ignored by git)
```

---

## 🚀 Models Implemented

### 1. Random Forest (TF-IDF + Numeric Features)
- **Vectorization:** TF-IDF with `max_features=5000`
- **Additional features:** `Sentence_Length` (character count) and `Word_Count`
- **Hyperparameters:** Optimized via `RandomizedSearchCV` (n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=2)
- **Test Accuracy:** ~54%

### 2. Bi-LSTM (Deep Learning)
- **Tokenizer:** `max_words=10000`, `max_len=100`, OOV token support
- **Architecture:** Embedding(128) → Bidirectional LSTM(64, return_sequences=True) → Dropout(0.5) → Bidirectional LSTM(32) → Dropout(0.5) → Dense(1, sigmoid)
- **Test Accuracy:** ~52%

> **Note:** Models show modest accuracy on the test set. Predictions are for demonstration purposes only — not financial advice.

---

## 💻 How to Run Locally

### Prerequisites
- Python 3.9+
- `pip` or `venv`

### Step 1: Clone the repository
```bash
git clone https://github.com/devpatel0005/TechMicra-AIML-Summer-Internship.git
cd TechMicra-AIML-Summer-Internship
```

### Step 2: Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# or
.venv\Scripts\activate           # Windows
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8507` (or the next available port).

---

## ☁️ Deploying on Streamlit Cloud

This app is ready for deployment on [Streamlit Cloud](https://streamlit.io/cloud).

### Deployment Steps
1. Push this repository to GitHub (main branch).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click "New app" and select:
   - **Repository:** `devpatel0005/TechMicra-AIML-Summer-Internship`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**.

### No Live Training Required
The UI loads pre-trained models from the `models/` folder. No model training happens at runtime, so the app starts instantly without GPU/CPU-intensive training.

### Model Files Included
All pre-trained model artifacts are committed to the repository under `models/`:
- `rf_model.pkl`, `tfidf_vectorizer.pkl`, `rf_metrics.json`
- `bilstm_model.keras`, `tokenizer.pkl`, `lstm_metrics.json`

---

## 🖥️ UI Features

| Feature | Description |
|---------|-------------|
| **Model Selection** | Choose between Random Forest or Bi-LSTM via sidebar radio |
| **Sentiment Prediction** | Enter a financial news sentence and get Positive/Negative prediction with confidence score |
| **Sample News Dropdown** | 6 pre-written realistic financial headlines for quick testing |
| **Clear Button** | Reset the text input instantly |
| **Model Evaluation Metrics** | View pre-computed test accuracy, classification report, and confusion matrix (no live testing) |
| **Dataset EDA** | View sentiment distribution, sentence length, and word count visualizations |
| **Attribution** | "Built by Dev Patel — 23AIML047" displayed at top left |

### Sample Headlines Included
1. Apple reports record quarterly revenue as iPhone sales exceed expectations
2. Tesla stock plunges after CEO announces production delays and missed delivery targets
3. Federal Reserve signals potential interest rate cut amid cooling inflation data
4. Amazon shares surge following strong AWS cloud computing growth announcement
5. Regional bank shares tumble on reports of significant loan defaults and credit losses
6. Pharmaceutical stock soars after successful phase 3 clinical trial results for new drug

---

## 🛠️ Training Scripts (Offline)

If you need to re-train the models:

### Full Training (Random Forest + Bi-LSTM)
```bash
python train_and_save_models.py
```
- Trains both models on the full dataset (~108k rows)
- Bi-LSTM trains for 5 epochs (may take 25+ minutes on CPU)
- Saves all model artifacts and metrics to `models/`

### Fast Bi-LSTM Training (Sample)
```bash
python train_bilstm_fast.py
```
- Trains Bi-LSTM on a stratified sample of 30,000 rows (3 epochs)
- Completes in ~2–3 minutes on CPU
- Saves Bi-LSTM model, tokenizer, and metrics to `models/`

---

## 📊 Dataset

- **File:** `Datasets/Sentiment_Stock_data.csv`
- **Rows:** 108,750 sentences
- **Columns:** `Sentiment` (0=Negative, 1=Positive), `Sentence`, `Sentence_Length`, `Word_Count`
- **Sentiment Distribution:** Balanced (~55,724 Positive, ~53,026 Negative)

The dataset is loaded from the local `Datasets/` folder, with a fallback to the GitHub raw URL if the file is missing.

---

## 👨‍💻 Author

**Built by Dev Patel**  
Student ID: **23AIML047**  
TechMicra AI/ML Summer Internship

---

## 📄 License

This project is part of the TechMicra AI/ML Summer Internship.
