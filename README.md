# 📩 SMS/Email Spam Classifier

A machine learning web app that classifies text messages as **Spam** or **Not Spam**, built with **Scikit-learn** and **Streamlit**.

The app trains and compares four classic ML models on a TF-IDF representation of the text, then lets you type in a message and get an instant prediction with a confidence score.

---

## 📑 Table of Contents

- [Features](#-features)
- [Screenshots](#️-screenshots)
- [Project Structure](#️-project-structure)
- [Requirements](#️-requirements)
- [Getting Started](#-getting-started)
- [How It Works](#-how-it-works)
- [Model Evaluation](#-model-evaluation)
- [Dataset](#-dataset)
- [Acknowledgements](#-acknowledgements)

---

## ✨ Features

- **Text preprocessing** — lowercasing, punctuation stripping, whitespace normalization
- **TF-IDF vectorization** — unigrams + bigrams, top 5,000 features, English stop words removed
- **Four trained classifiers**, compared side by side:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Linear SVM
  - Random Forest
- **Evaluation metrics** for each model — accuracy, precision, recall, F1-score, confusion matrix, and full classification report
- **Live prediction** on any custom message, with confidence (for probabilistic models) or decision score (for SVM)
- **Cached data loading & training** via Streamlit's `@st.cache_data` / `@st.cache_resource` for fast reloads

---

## 🖼️ Screenshots

**Classify Message** — live spam detector with quick example buttons
<img width="1919" height="948" alt="image" src="https://github.com/user-attachments/assets/433a0a9a-d4f3-483d-b5c7-c14155c6301d" />


**Model Metrics** — accuracy, precision, recall, F1, confusion matrix & classification report per model
<img width="1919" height="937" alt="image" src="https://github.com/user-attachments/assets/8fe8ef6d-d4ba-4ccc-9be4-9445afc8329e" />


**Model Comparison** — side-by-side comparison of all four classifiers
<img width="1915" height="947" alt="image" src="https://github.com/user-attachments/assets/f44ba5e4-2ced-4b88-8764-e7e524490372" />


**Dataset Overview** — label distribution and sample messages
<img width="1917" height="932" alt="image" src="https://github.com/user-attachments/assets/f34f5b25-3242-46a3-bae9-01be6e7af52c" />


**Model Selector** — switch between classifiers from the sidebar
<img width="415" height="404" alt="image" src="https://github.com/user-attachments/assets/c03cffc3-3c48-4fd9-b300-fc3ff2566a0f" />


---

## 🗂️ Project Structure

```
.
├── app.py       # Streamlit UI (entry point)
├── code.py      # ML backend: preprocessing, data loading, training, prediction
└── spam.csv     # Dataset (SMS Spam Collection format: v1=label, v2=message)
```

---

## ⚙️ Requirements

- Python 3.8+
- pandas
- scikit-learn
- streamlit

Install dependencies:

```bash
pip install pandas scikit-learn streamlit
```

*(Consider adding a `requirements.txt` with pinned versions for reproducibility.)*

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Add the dataset**
   Place `spam.csv` in the project root. The file should contain at least two columns:
   - `v1` — the label (`spam` or `ham`)
   - `v2` — the raw message text

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. Open the local URL Streamlit prints (usually `http://localhost:8501`) in your browser.

---

## 🧠 How It Works

1. **`preprocess(text)`** — cleans raw text (lowercase, strip punctuation, collapse whitespace).
2. **`load_data(path)`** — loads `spam.csv`, drops missing rows, applies preprocessing, and encodes labels (`spam` → 1, `ham` → 0). Cached with `st.cache_data`.
3. **`train_models(df)`** — splits data 80/20 (stratified), fits a `TfidfVectorizer`, and trains all four classifiers. Cached with `st.cache_resource` so training only happens once per session.
4. **`predict_message(model, vectorizer, raw_text)`** — preprocesses and vectorizes a new message, runs it through the chosen model, and returns the prediction, a display label, and a confidence/decision score.

---

## 📊 Model Evaluation

Each model's results include:

| Metric | Description |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | Of predicted spam, how much was actually spam |
| Recall | Of actual spam, how much was caught |
| F1-score | Harmonic mean of precision & recall |
| Confusion Matrix | True/false positives & negatives |

These are computed automatically for every model during training and can be surfaced in the Streamlit UI for comparison.

---

## 📁 Dataset

This project expects the classic **SMS Spam Collection** CSV format (`v1`, `v2` columns, `latin-1` encoding). You can find similar datasets on [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

---

## 🙌 Acknowledgements

Built with [Streamlit](https://streamlit.io/) and [scikit-learn](https://scikit-learn.org/).
