"""
code.py — ML Backend
=====================
Handles all machine learning logic:
  - Text preprocessing
  - Data loading
  - TF-IDF vectorisation
  - Training multiple Scikit-learn classifiers
  - Prediction helper

Imported by app.py (Streamlit UI).
Run the app with: streamlit run app.py
"""

import os
import re
import string
import warnings

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Only silence the noisy convergence warnings we expect from LinearSVC /
# LogisticRegression on sparse TF-IDF data — don't blanket-suppress everything.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ──────────────────────────────────────────────
# 1. Text Preprocessing
# ──────────────────────────────────────────────
def preprocess(text: str) -> str:
    """Lowercase → strip URLs/numbers → strip punctuation → collapse whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)          # normalise URLs
    text = re.sub(r"\b\d{5,}\b", " LONGNUM ", text)             # long digit runs (phone/premium numbers)
    text = re.sub(r"\b\d+\b", " NUM ", text)                    # remaining standalone numbers
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ──────────────────────────────────────────────
# 2. Data Loading  (cached so it runs once)
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path: str = "spam.csv") -> pd.DataFrame:
    """Load spam.csv, clean it, and return a ready DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Make sure the dataset is in the working "
            f"directory (expects the classic Kaggle SMS Spam Collection CSV "
            f"with 'v1' = label, 'v2' = message)."
        )

    try:
        df = pd.read_csv(path, encoding="latin-1", usecols=["v1", "v2"])
    except ValueError as e:
        raise ValueError(
            f"'{path}' doesn't have the expected columns ('v1', 'v2'). "
            f"Got columns: {pd.read_csv(path, encoding='latin-1', nrows=0).columns.tolist()}"
        ) from e
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            e.encoding, e.object, e.start, e.end,
            f"Failed to decode '{path}' as latin-1. Check the file's encoding."
        ) from e

    df.columns = ["label", "message"]
    df.dropna(subset=["label", "message"], inplace=True)

    if df.empty:
        raise ValueError(f"'{path}' loaded but contains no valid rows after cleaning.")

    df["clean_message"] = df["message"].apply(preprocess)
    df["label_enc"] = df["label"].map({"spam": 1, "ham": 0})

    unmapped = df["label_enc"].isna().sum()
    if unmapped:
        st.warning(
            f"Dropped {unmapped} row(s) with unrecognised labels "
            f"(expected 'spam' or 'ham')."
        )
        df.dropna(subset=["label_enc"], inplace=True)
    df["label_enc"] = df["label_enc"].astype(int)

    return df


# ──────────────────────────────────────────────
# 3. Model Training  (cached so it runs once)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    """
    Vectorise text with TF-IDF and train four classifiers.

    Returns
    -------
    vectorizer : TfidfVectorizer  (fitted)
    results    : dict  — per-model metrics and trained model objects
    y_test     : Series  — ground-truth test labels
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"],
        df["label_enc"],
        test_size=0.20,
        random_state=42,
        stratify=df["label_enc"],
    )

    # TF-IDF vectorisation (unigrams + bigrams, top 5 000 features)
    vectorizer = TfidfVectorizer(
        max_features=5_000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_tr = vectorizer.fit_transform(X_train)
    X_te = vectorizer.transform(X_test)

    # Classifiers to compare
    classifiers = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression":     LogisticRegression(
            max_iter=1000, C=5, solver="lbfgs", n_jobs=-1
        ),
        "Linear SVM":              LinearSVC(C=1.0, max_iter=2000),
        "Random Forest":           RandomForestClassifier(
            n_estimators=100,
            max_depth=50,        # cap depth — sparse TF-IDF trees overfit/slow down otherwise
            n_jobs=-1,            # use all cores
            random_state=42,
        ),
    }

    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_tr, y_train)
        preds = clf.predict(X_te)
        results[name] = {
            "model":     clf,
            "accuracy":  accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall":    recall_score(y_test, preds),
            "f1":        f1_score(y_test, preds),
            "cm":        confusion_matrix(y_test, preds),
            "report":    classification_report(y_test, preds, target_names=["Not Spam", "Spam"]),
        }

    return vectorizer, results, y_test


# ──────────────────────────────────────────────
# 4. Prediction Helper
# ──────────────────────────────────────────────
def predict_message(model, vectorizer, raw_text: str):
    """
    Classify a single raw message as 'spam' or 'not spam'.

    Parameters
    ----------
    model       : fitted sklearn classifier
    vectorizer  : fitted TfidfVectorizer
    raw_text    : str — the original message

    Returns
    -------
    prediction  : int   — 1 (spam) or 0 (not spam)
    label       : str   — "SPAM 🔴" or "NOT SPAM 🟢"
    badge_class : str   — CSS class name for the result badge
    conf_html   : str   — HTML snippet showing confidence / decision score
    clean_text  : str   — preprocessed version of the input
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Input text is empty — nothing to classify.")

    clean_text = preprocess(raw_text)
    vec        = vectorizer.transform([clean_text])
    prediction = model.predict(vec)[0]

    label       = "SPAM 🔴"      if prediction == 1 else "NOT SPAM 🟢"
    badge_class = "spam-badge"  if prediction == 1 else "not-spam-badge"

    conf_html = ""
    if hasattr(model, "predict_proba"):
        proba     = model.predict_proba(vec)[0]
        conf      = proba[prediction] * 100
        conf_html = f'<p style="color:#ccc;margin-top:.5rem;">Confidence: <b>{conf:.1f}%</b></p>'
    elif hasattr(model, "decision_function"):
        score     = abs(model.decision_function(vec)[0])
        conf_html = f'<p style="color:#ccc;margin-top:.5rem;">Decision score: <b>{score:.2f}</b></p>'

    return prediction, label, badge_class, conf_html, clean_text