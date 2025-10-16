# app.py — Apply-Insight Portal
from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st

# Optional ML bits for the Technical tab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ======================
# Paths & data loading
# ======================
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "decisions_enriched.csv"  # make sure this file exists in your repo

st.set_page_config(page_title="Apply-Insight Portal", layout="wide")


@st.cache_data(show_spinner=False)
def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        # Helpful error with directory listing
        raise FileNotFoundError(
            f"Missing data file: {path}\n"
            f"Existing files in {DATA_DIR}: {[p.name for p in DATA_DIR.glob('*')]}"
        )
    df = pd.read_csv(path)
    # normalize core fields
    df["would_apply"] = df["would_apply"].astype(str).str.lower().str.strip()
    df = df[df["would_apply"].isin(["yes", "no"])].copy()
    df["label"] = (df["would_apply"] == "yes").astype(int)
    df["rationale"] = df["rationale"].fillna("").astype(str)
    if "profile_text" not in df.columns:
        # derive a simple profile_text if not present
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


def norm_tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9\-]+", str(s).lower())


@st.cache_data(show_spinner=False)
def compute_kpis(df: pd.DataFrame) -> dict:
    yes_pct = 100 * df["label"].mean()
    no_pct = 100 - yes_pct
    avg_len = df["rationale"].apply(lambda s: len(str(s).split())).mean()
    auto_fixed_pct = 100 * df["auto_fixed"].astype(str).str.lower().eq("yes").mean()
    yes_by_bg = (df.groupby("academic_background")["label"].mean().mul(100).sort_values(ascending=False))
    yes_by_exp = (df.groupby("previous_work_experience")["label"].mean().mul(100).sort_index())
    return {
        "yes_pct": round(yes_pct, 1),
        "no_pct": round(no_pct, 1),
        "avg_len": round(avg_len, 1),
        "auto_fixed_pct": round(auto_fixed_pct, 1),
        "yes_by_bg": yes_by_bg,
        "yes_by_exp": yes_by_exp,
    }


@st.cache_data(show_spinner=False)
def compute_overlap(df: pd.DataFrame) -> pd.Series:
    def overlap_row(r):
        p = set(norm_tokens(r["profile_text"]))
        ra = set(norm_tokens(r["rationale"]))
        return sorted(p & ra)
    return df.apply(lambda r: overlap_row(r), axis=1)


@st.cache_data(show_spinner=False)
def tfidf_and_drivers(
    df: pd.DataFrame,
    ngrams=(1, 2),
    min_df=3,
    top_n=30,
):
    """Returns (top_yes_terms_df, top_no_terms_df, drivers_yes_df, drivers_no_df, holdout_acc)."""
    texts = df["profile_text"].astype(str).values
    y = df["label"].values

    tfidf = TfidfVectorizer(ngram_range=ngrams, min_df=min_df, stop_words="english")
    try:
        X = tfidf.fit_transform(texts)
    except ValueError as e:
        # happens if vocabulary is empty / dataset tiny
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0.0)

    terms = np.array(tfidf.get_feature_names_out())

    # Class signature terms (Yes vs No) via mean TF-IDF difference
    yes_mean = X[y == 1].mean(axis=0)
    no_mean = X[y == 0].mean(axis=0)
    diff = np.asarray(yes_mean - no_mean).ravel()

    top_yes_idx = diff.argsort()[::-1][:top_n]
    top_no_idx = diff.argsort()[:top_n]

    top_yes_terms = pd.DataFrame({"term": terms[top_yes_idx], "score": diff[top_yes_idx]})
    top_no_terms = pd.DataFrame({"term": terms[top_no_idx], "score": -diff[top_no_idx]})

    # Lightweight supervised drivers: Logistic Regression coefficients
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        clf = LogisticRegression(max_iter=400)
        clf.fit(X_train, y_train)
        acc = float(clf.score(X_test, y_test))
        coef = clf.coef_.ravel()

        drivers_yes_idx = np.argsort(coef)[-top_n:][::-1]
        drivers_no_idx = np.argsort(coef)[:top_n]

        drivers_yes = pd.DataFrame({"term": terms[drivers_yes_idx], "weight": coef[drivers_yes_idx]})
        drivers_no = pd.DataFrame({"term": terms[drivers_no_idx], "weight": coef[drivers_no_idx]})
    except Exception:
        # if tiny data or singular matrix, degrade gracefully
        acc = 0.0
        drivers_yes = pd.DataFrame()
        drivers_no = pd.DataFrame()

    return (top_yes_terms, top_no_terms, drivers_yes, drivers_no, acc)


# ======================
# UI
# ======================
st.title("🎓 Apply-Insight Portal")

# Load data (with friendly error if missing)
try:
    df = load_df(CSV_PATH)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Tabs
tab_overview, tab_lab = st.tabs(["Overview (Non-Technical)", "Lab (Technical)"])

# ----------------------
# Overview
# ----------------------
with tab_overview:
    kpis = compute_kpis(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Yes %", f"{kpis['yes_pct']:.1f}%")
    c2.metric("No %", f"{kpis['no_pct']:.1f}%")
    c3.metric("Avg rationale words", f"{kpis['avg_len']:.1f}")
    c4.metric("% auto-fixed", f"{kpis['auto_fixed_pct']:.1f}%")

    st.subheader("Yes % by Academic Background")
    st.bar_chart(kpis["yes_by_bg"].sort_values().rename("Yes %"))

    st.subheader("Yes % by Work Experience")
    st.bar_chart(kpis["yes_by_exp"].rename("Yes %"))

    st.subheader("Filter & Explore")
    colf1, colf2 = st.columns(2)
    bg_sel = colf1.multiselect(
        "Academic background",
        sorted(df["academic_background"].astype(str).unique().tolist()),
    )
    exp_sel = colf2.multiselect(
        "Previous work experience",
        sorted(df["previous_work_experience"].astype(str).unique().tolist()),
    )
    view = df.copy()
    if bg_sel:
        view = view[view["academic_background"].isin(bg_sel)]
    if exp_sel:
        view = view[view["previous_work_experience"].isin(exp_sel)]
    st.dataframe(view, use_container_width=True)

    st.download_button(
        "Download current view (CSV)",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="decisions_view.csv",
        mime="text/csv",
    )

# ----------------------
# Technical Lab
# ----------------------
with tab_lab:
    st.caption("Deep dive: token drivers, class signatures, and overlap with rationale.")
    with st.expander("Raw data (first 100 rows)"):
        st.dataframe(df.head(100), use_container_width=True)

    # Overlap tokens (profile ↔ rationale)
    st.subheader("Profile ↔ Rationale Overlap")
    df = df.copy()
    if "overlap_tokens" not in df.columns:
        df["overlap_tokens"] = compute_overlap(df)
    df["overlap_count"] = df["overlap_tokens"].apply(len)
    st.write("Average overlap tokens by decision:",
             df.groupby("would_apply")["overlap_count"].mean().round(2).to_dict())
    st.dataframe(df[["academic_background","previous_work_experience","would_apply","overlap_tokens","overlap_count"]]
                 .sort_values("overlap_count", ascending=False)
                 .head(30),
                 use_container_width=True)

    # Token drivers & signatures
    st.subheader("Keyword Signals")
    colcfg1, colcfg2, colcfg3 = st.columns(3)
    ngram = colcfg1.selectbox("n-grams", options=["1", "1-2"], index=1)
    ngrams = (1, 2) if ngram == "1-2" else (1, 1)
    min_df = colcfg2.slider("min_df (ignore rare tokens)", 1, 10, 3, 1)
    top_n = colcfg3.slider("Top-N terms", 10, 50, 30, 5)

    top_yes, top_no, drv_yes, drv_no, acc = tfidf_and_drivers(df, ngrams=ngrams, min_df=min_df, top_n=top_n)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Characteristic of YES (TF-IDF diff)**")
        st.dataframe(top_yes.head(top_n), use_container_width=True)
    with col2:
        st.write("**Characteristic of NO (TF-IDF diff)**")
        st.dataframe(top_no.head(top_n), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.write("**Push toward YES (LogReg coefficients)**")
        st.dataframe(drv_yes.head(top_n), use_container_width=True)
    with col4:
        st.write("**Push toward NO (LogReg coefficients)**")
        st.dataframe(drv_no.head(top_n), use_container_width=True)

    st.caption(f"Logistic Regression holdout accuracy (sanity check, not a product model): **{acc:.3f}**")

    # Per-profile mini explain (token list)
    st.subheader("Per-Profile Token Explain (quick look)")
    idx = st.number_input("Row index", min_value=0, max_value=max(0, len(df)-1), value=0, step=1)
    row = df.iloc[int(idx)]
    st.write("**Would apply:**", row["would_apply"])
    st.write("**Rationale:**")
    st.code(row["rationale"])
    st.write("**Overlap tokens:**", ", ".join(row["overlap_tokens"]) if row["overlap_tokens"] else "—")

st.caption("Data source: data/decisions_enriched.csv • This app computes indicative (not causal) signals.")
