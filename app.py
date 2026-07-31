"""
app.py — Streamlit UI
======================
All interface logic for the Spam SMS Classifier.
ML backend is imported from code.py.

Run with: streamlit run app.py
"""

import pandas as pd
import streamlit as st

from code import load_data, predict_message, train_models

# ══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Spam SMS Classifier",
    page_icon="🚫",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════
#  GLOBAL STYLES
# ══════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    /* ---------- App background ---------- */
    .stApp { background: #0f1117; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d27 0%, #12151f 100%);
        border-right: 1px solid #2a2d3e;
    }

    /* ---------- Metric cards ---------- */
    [data-testid="stMetric"] {
        background: #1a1d27;
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 1rem;
    }
    [data-testid="stMetricValue"] { color: #7c6af7 !important; font-size: 2rem !important; }
    [data-testid="stMetricLabel"] { color: #8b8fa8 !important; }

    /* ---------- Prediction badges ---------- */
    .spam-badge {
        background: linear-gradient(135deg, #ff4b6e, #ff2244);
        color: white; font-size: 1.6rem; font-weight: 800;
        padding: 1.2rem 2rem; border-radius: 16px; text-align: center;
        box-shadow: 0 0 30px rgba(255,68,100,.4);
        animation: pulse 1.5s ease-in-out infinite;
    }
    .not-spam-badge {
        background: linear-gradient(135deg, #00c896, #009e72);
        color: white; font-size: 1.6rem; font-weight: 800;
        padding: 1.2rem 2rem; border-radius: 16px; text-align: center;
        box-shadow: 0 0 30px rgba(0,200,150,.4);
        animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse {
        0%,100% { transform: scale(1); }
        50%      { transform: scale(1.03); }
    }

    /* ---------- Section headers ---------- */
    .section-title {
        color: #7c6af7; font-size: 1.15rem; font-weight: 700;
        letter-spacing: .06em; text-transform: uppercase;
        border-left: 4px solid #7c6af7; padding-left: .6rem;
        margin-bottom: 1rem;
    }

    /* ---------- DataFrame ---------- */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* ---------- Tabs ---------- */
    .stTabs [role="tablist"] { border-bottom: 2px solid #2a2d3e; }
    .stTabs [role="tab"][aria-selected="true"] {
        color: #7c6af7; border-bottom: 2px solid #7c6af7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚫 Spam SMS Classifier")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    selected_model = st.selectbox(
        "Classification Model",
        ["Multinomial Naive Bayes", "Logistic Regression", "Linear SVM", "Random Forest"],
        index=0,
    )

    st.markdown("---")
    st.markdown(
        """
        **Pipeline**
        - 📥 Load `spam.csv`
        - 🔤 Text Preprocessing
        - 📊 TF-IDF Vectorisation
        - 🤖 Multiple ML Classifiers
        - 🎛️ Interactive Streamlit UI
        """
    )
    st.markdown("---")
    st.caption("Built with Scikit-learn & Streamlit")


# ══════════════════════════════════════════════════════════
#  LOAD DATA & TRAIN  (delegated to code.py)
# ══════════════════════════════════════════════════════════
with st.spinner("⏳ Loading data & training models…"):
    df                       = load_data()
    vectorizer, results, _   = train_models(df)

chosen = results[selected_model]
model  = chosen["model"]


# ══════════════════════════════════════════════════════════
#  HERO HEADER
# ══════════════════════════════════════════════════════════
st.markdown(
    """
    <div style="text-align:center; padding: 2rem 0 1rem;">
      <h1 style="font-size:2.8rem; font-weight:900;
                 background:linear-gradient(135deg,#7c6af7,#ff4b6e);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        🚫 Spam SMS Classifier
      </h1>
      <p style="color:#8b8fa8; font-size:1.05rem; margin-top:-.4rem;">
        Machine Learning · TF-IDF Vectorisation · Scikit-learn · Streamlit
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Classify Message", "📊 Model Metrics", "📈 Model Comparison", "📋 Dataset"]
)


# ──────────────────────────────────────────────
# TAB 1 · LIVE CLASSIFIER
# ──────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-title">Live Spam Detector</p>', unsafe_allow_html=True)

    col_input, col_result = st.columns([3, 2], gap="large")

    with col_input:
        user_msg = st.text_area(
            "Enter an SMS / email message:",
            height=200,
            placeholder="Type or paste a message here…",
        )

        ex_col1, ex_col2 = st.columns(2)
        with ex_col1:
            if st.button("🔴 Try Spam Example"):
                user_msg = "WINNER!! You've been selected to receive a £900 prize reward! Call 09061701461 now. Claim code: KL341. Valid 12 hrs only."
        with ex_col2:
            if st.button("🟢 Try Not Spam Example"):
                user_msg = "Hey, are we still meeting for lunch tomorrow at 1pm?"

        classify_btn = st.button("🚀 Classify", use_container_width=True, type="primary")

    with col_result:
        if classify_btn and user_msg.strip():
            # Prediction delegated entirely to code.py
            _, label, badge, conf_html, clean = predict_message(model, vectorizer, user_msg)

            st.markdown(
                f'<div class="{badge}">{label}</div>{conf_html}',
                unsafe_allow_html=True,
            )
            st.markdown("---")
            st.markdown(f"**Model used:** `{selected_model}`")
            st.markdown(f"**Preprocessed:** _{clean[:120]}{'…' if len(clean) > 120 else ''}_")

        elif classify_btn:
            st.warning("Please enter a message first.")


# ──────────────────────────────────────────────
# TAB 2 · MODEL METRICS
# ──────────────────────────────────────────────
with tab2:
    st.markdown(
        f'<p class="section-title">Metrics — {selected_model}</p>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("✅ Accuracy",  f"{chosen['accuracy']  * 100:.2f}%")
    m2.metric("🎯 Precision", f"{chosen['precision'] * 100:.2f}%")
    m3.metric("📡 Recall",    f"{chosen['recall']    * 100:.2f}%")
    m4.metric("⚖️ F1 Score",  f"{chosen['f1']        * 100:.2f}%")

    st.markdown("#### Confusion Matrix")
    cm_df = pd.DataFrame(
        chosen["cm"],
        index=["Actual Not Spam", "Actual Spam"],
        columns=["Predicted Not Spam", "Predicted Spam"],
    )
    st.dataframe(cm_df.style.background_gradient(cmap="Purples"), use_container_width=True)

    st.markdown("#### Classification Report")
    st.code(chosen["report"], language="text")


# ──────────────────────────────────────────────
# TAB 3 · MODEL COMPARISON
# ──────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-title">All Models — Side-by-side Comparison</p>', unsafe_allow_html=True)

    rows = [
        {
            "Model":     name,
            "Accuracy":  f"{r['accuracy']  * 100:.2f}%",
            "Precision": f"{r['precision'] * 100:.2f}%",
            "Recall":    f"{r['recall']    * 100:.2f}%",
            "F1 Score":  f"{r['f1']        * 100:.2f}%",
        }
        for name, r in results.items()
    ]
    comp_df    = pd.DataFrame(rows)
    best_model = max(results, key=lambda k: results[k]["f1"])

    def highlight_best(row):
        style = "background-color: #2a1f4a; color: #b39dfd; font-weight:700"
        return [style if row["Model"] == best_model else ""] * len(row)

    st.dataframe(
        comp_df.style.apply(highlight_best, axis=1),
        use_container_width=True,
        hide_index=True,
    )
    st.success(f"🏆 Best model by F1 Score: **{best_model}**")

    chart_df = pd.DataFrame(
        {
            "Model":    list(results.keys()),
            "Accuracy": [r["accuracy"] * 100 for r in results.values()],
            "F1 Score": [r["f1"]       * 100 for r in results.values()],
        }
    ).set_index("Model")
    st.markdown("#### Accuracy & F1 Score")
    st.bar_chart(chart_df, color=["#7c6af7", "#ff4b6e"])


# ──────────────────────────────────────────────
# TAB 4 · DATASET EXPLORER
# ──────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-title">Dataset Overview</p>', unsafe_allow_html=True)

    total = len(df)
    spam  = (df["label"] == "spam").sum()
    ham   = total - spam

    d1, d2, d3 = st.columns(3)
    d1.metric("📩 Total Messages", f"{total:,}")
    d2.metric("🔴 Spam",           f"{spam:,}  ({spam / total * 100:.1f}%)")
    d3.metric("🟢 Not Spam",       f"{ham:,}  ({ham  / total * 100:.1f}%)")

    st.markdown("#### Label Distribution")
    dist_df = pd.DataFrame({"Count": {"Not Spam": ham, "Spam": spam}})
    st.bar_chart(dist_df, color=["#7c6af7"])

    st.markdown("#### Sample Messages")
    filter_label = st.radio("Filter by:", ["All", "Not Spam", "Spam"], horizontal=True)
    view = (
        df if filter_label == "All"
        else df[df["label"] == ("ham" if filter_label == "Not Spam" else "spam")]
    )
    st.dataframe(
        view[["label", "message"]].head(50).reset_index(drop=True),
        use_container_width=True,
        height=350,
    )
