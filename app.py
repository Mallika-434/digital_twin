# app.py — Student Apply-Insight Portal (Final Version with Graph Grouping + Tooltips + Number Input)

from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# 1️⃣ Page setup
# -------------------------------------------------
st.set_page_config(page_title="Student Apply-Insight Portal", layout="wide")
st.title("🎓 Student Apply-Insight Portal")

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "decisions_enriched.csv"

# -------------------------------------------------
# 2️⃣ Load dataset
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"❌ File not found: {path}")
        st.write("Available files in /data:", [p.name for p in DATA_DIR.glob('*')])
        st.stop()
    df = pd.read_csv(path)
    df["would_apply"] = df["would_apply"].astype(str).str.lower().str.strip()
    df = df[df["would_apply"].isin(["yes", "no"])].copy()
    df["label"] = (df["would_apply"] == "yes").astype(int)
    df["rationale"] = df["rationale"].fillna("").astype(str)
    if "auto_fixed" in df.columns:
        df.drop(columns=["auto_fixed"], inplace=True)
    if "profile_text" not in df.columns:
        def build_profile_text(r):
            return (
                f"{r.get('academic_background','')}. "
                f"{r.get('academic_interests','')}. "
                f"{r.get('professional_interests','')}. "
                f"gender:{r.get('gender','')}. "
                f"age:{r.get('age','')}. "
                f"work:{r.get('previous_work_experience','')}."
            )
        df["profile_text"] = df.apply(build_profile_text, axis=1)
    return df

df = load_df(CSV_PATH)

# -------------------------------------------------
# 3️⃣ Shared helper: fit model for probabilities
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def fit_model_for_probs(df_in: pd.DataFrame, ngrams=(1, 2), min_df=3):
    try:
        y = df_in["would_apply"].astype(str).eq("yes").astype(int)
        tfidf = TfidfVectorizer(ngram_range=ngrams, min_df=min_df, stop_words="english")
        X = tfidf.fit_transform(df_in["profile_text"])
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        clf = LogisticRegression(max_iter=400)
        clf.fit(X_tr, y_tr)
        acc = float(clf.score(X_te, y_te))
        prob_yes = clf.predict_proba(X)[:, 1]
        return prob_yes, tfidf, clf, acc
    except Exception:
        return None, None, None, None

prob_yes, shared_vec, shared_clf, shared_acc = fit_model_for_probs(df)
if prob_yes is not None:
    df["prob_yes"] = prob_yes
    df["prob_no"] = 1 - prob_yes

# -------------------------------------------------
# 4️⃣ Sidebar controls (with tooltips)
# -------------------------------------------------
st.sidebar.header("Interactive Controls")

thr = st.sidebar.slider(
    "Decision threshold (P(YES) ≥ …)",
    0.0, 1.0, 0.50, 0.01,
    help="Set how confident the model must be before classifying a profile as 'YES'."
)

use_predictions = st.sidebar.toggle(
    "Use model predictions instead of original Yes/No",
    value=False,
    help="Switch between the LLM's original answers and the logistic model's predictions."
)

if "prob_yes" in df.columns:
    df["predicted_apply"] = np.where(df["prob_yes"] >= thr, "yes", "no")
    df["pred_label"] = (df["predicted_apply"] == "yes").astype(int)
else:
    df["predicted_apply"] = df["would_apply"]
    df["pred_label"] = df["label"]

# -------------------------------------------------
# 5️⃣ Helper functions
# -------------------------------------------------
def _norm_tokens(s): return re.findall(r"[a-z0-9\-]+", str(s).lower())

@st.cache_data(show_spinner=False)
def compute_overlap_cols(df_):
    df2 = df_.copy()
    df2["overlap_tokens"] = df2.apply(
        lambda r: sorted(set(_norm_tokens(r["profile_text"])) & set(_norm_tokens(r["rationale"]))),
        axis=1
    )
    df2["overlap_count"] = df2["overlap_tokens"].apply(len)
    return df2

@st.cache_data(show_spinner=False)
def tfidf_and_drivers(texts, labels, ngrams=(1, 2), min_df=3, top_n=30):
    tfidf = TfidfVectorizer(ngram_range=ngrams, min_df=min_df, stop_words="english")
    X = tfidf.fit_transform(texts)
    terms = np.array(tfidf.get_feature_names_out())
    y = np.asarray(labels)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=400)
    clf.fit(X_tr, y_tr)
    acc = float(clf.score(X_te, y_te))
    coef = clf.coef_.ravel()
    drv_yes = pd.DataFrame({"term": terms[np.argsort(coef)[-top_n:][::-1]], "weight": np.sort(coef)[-top_n:][::-1]})
    drv_no = pd.DataFrame({"term": terms[np.argsort(coef)[:top_n]], "weight": np.sort(coef)[:top_n]})
    return drv_yes, drv_no, acc, tfidf, clf

def summarize_group(df, group_col):
    g = df.groupby(group_col).agg(rows=("label", "size"), yes_rate=("pred_label", "mean"), avg_prob_yes=("prob_yes", "mean")).reset_index()
    g["yes_rate"] = (g["yes_rate"] * 100).round(2)
    g["avg_prob_yes"] = g["avg_prob_yes"].round(3)
    return g.sort_values("avg_prob_yes", ascending=False)

# -------------------------------------------------
# 6️⃣ Tabs
# -------------------------------------------------
tab_overview, tab_tech = st.tabs(["Non-Technical", "Technical"])

# =================================================
# 🟢 NON-TECHNICAL TAB
# =================================================
with tab_overview:
    # KPIs
    base_label = "pred_label" if use_predictions else "label"
    yes_pct = 100 * df[base_label].mean()
    no_pct = 100 - yes_pct
    avg_len = df["rationale"].apply(lambda s: len(str(s).split())).mean()

    c1, c2, c3 = st.columns(3)
    c1.metric(("Predicted Yes %" if use_predictions else "Yes %"), f"{yes_pct:.1f}%")
    c2.metric(("Predicted No %" if use_predictions else "No %"), f"{no_pct:.1f}%")
    c3.metric("Avg rationale words", f"{avg_len:.1f}")
    st.markdown("---")

    # -------------------------------
    # Decision Insights section
    # -------------------------------
    with st.container():
        st.subheader("📊 Decision Insights")

        # Pie chart
        dec_col = "predicted_apply" if use_predictions else "would_apply"
        yes_count = int((df[dec_col] == "yes").sum())
        no_count = int((df[dec_col] == "no").sum())
        donut_df = pd.DataFrame({"Decision": ["Yes", "No"], "Count": [yes_count, no_count]})
        st.plotly_chart(
            px.pie(donut_df, names="Decision", values="Count", hole=0.55,
                   title="Predicted Would Apply — Yes vs No" if use_predictions else "Would Apply — Yes vs No"),
            use_container_width=True
        )

        # Bar chart 1: by background
        yes_by_bg = df.groupby("academic_background")[base_label].mean().mul(100).reset_index()
        yes_by_bg.rename(columns={base_label: "Yes %"}, inplace=True)
        st.plotly_chart(
            px.bar(yes_by_bg, x="academic_background", y="Yes %", color="Yes %",
                   color_continuous_scale="Blues", title="Yes % by Academic Background").update_layout(xaxis_tickangle=-45),
            use_container_width=True
        )

        # Bar chart 2: by experience
        yes_by_exp = df.groupby("previous_work_experience")[base_label].mean().mul(100).reset_index()
        yes_by_exp.rename(columns={base_label: "Yes %"}, inplace=True)
        st.plotly_chart(
            px.bar(yes_by_exp, x="previous_work_experience", y="Yes %", title="Work Experience Effect"),
            use_container_width=True
        )

    st.markdown("---")

    # -------------------------------
    # Filter & Explore
    # -------------------------------
    st.subheader("🔍 Filter & Explore")
    df["_background"] = df["academic_background"].astype(str)
    df["_experience"] = df["previous_work_experience"].astype(str)
    df["_apply"] = df["would_apply"].astype(str)

    bg_sel = st.multiselect(
        "Academic background",
        sorted(df["_background"].unique().tolist()),
        help="Filter data by academic background (e.g., Engineering, History, etc.)"
    )

    exp_sel = st.multiselect(
        "Previous work experience",
        sorted(df["_experience"].unique().tolist()),
        help="Filter data by whether users have prior work experience."
    )

    apply_sel = st.multiselect(
        "Would apply",
        ["yes", "no"],
        help="Filter data by model or LLM decision outcome."
    )

    view = df.copy()
    if bg_sel:
        view = view[view["_background"].isin(bg_sel)]
    if exp_sel:
        view = view[view["_experience"].isin(exp_sel)]
    if apply_sel:
        view = view[view["_apply"].isin(apply_sel)]

    display_cols = [c for c in view.columns if c not in ["_background", "_experience", "_apply"]]
    st.caption(f"Showing {len(view):,} of {len(df):,} rows")
    if "prob_yes" in view:
        styled = view[display_cols].style.format({"prob_yes": "{:.2f}"}).background_gradient(subset=["prob_yes"], cmap="Greens")
        st.dataframe(styled, use_container_width=True)
    else:
        st.dataframe(view[display_cols], width="stretch")

# =================================================
# 🧪 TECHNICAL TAB
# =================================================
with tab_tech:
    st.subheader("Deep Dive: Drivers, Overlap & Probabilities")

    df_overlap = compute_overlap_cols(df)
    avg_overlap = df_overlap.groupby("would_apply")["overlap_count"].mean().round(2).to_dict()
    st.write("Average overlap tokens by decision:", avg_overlap)
    st.dataframe(df_overlap[["academic_background", "previous_work_experience", "would_apply", "overlap_tokens", "overlap_count"]].head(40), width="stretch")

    st.markdown("---")

    st.markdown("### Logistic Regression Drivers (n-grams = (1, 2))")
    top_n = st.number_input(
        "Top-N terms",
        min_value=10,
        max_value=50,
        value=30,
        step=1,
        help="Enter how many top weighted terms you want to display (between 10 and 50)."
    )
    drv_yes, drv_no, acc, vec, clf = tfidf_and_drivers(df["profile_text"], df["label"], top_n=top_n)
    st.caption(f"Logistic holdout accuracy: **{acc:.3f}**")

    c3, c4 = st.columns(2)
    with c3:
        st.write("**Push toward YES**")
        st.dataframe(drv_yes.head(top_n), use_container_width=True)
    with c4:
        st.write("**Push toward NO**")
        st.dataframe(drv_no.head(top_n), use_container_width=True)

    if not drv_yes.empty:
        top10_yes = drv_yes.head(10).iloc[::-1]
        fig_drv = px.bar(top10_yes, x="weight", y="term", orientation="h", title="Top-10 YES Drivers")
        st.plotly_chart(fig_drv, use_container_width=True)

    st.markdown("---")

    st.markdown("### Model Probability (per profile)")
    if vec is not None and clf is not None:
        X_all = vec.transform(df["profile_text"])
        df["prob_yes"] = clf.predict_proba(X_all)[:, 1]
        df["prob_no"] = 1 - df["prob_yes"]

        st.subheader("Distribution of YES probabilities")
        fig_prob = px.histogram(df, x="prob_yes", nbins=20, title="Model confidence: P(YES)")
        fig_prob.update_layout(xaxis_title="P(YES)", yaxis_title="Count")
        st.plotly_chart(fig_prob, use_container_width=True)

st.caption("Data source: data/decisions_enriched.csv • Student Apply-Insight Portal")
