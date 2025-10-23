# app.py — Student Apply-Insight Portal (Interactive + Drivers-only)

from pathlib import Path
import re
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -----------------------------
# Page config & title
# -----------------------------
st.set_page_config(page_title="Student Apply-Insight Portal", layout="wide")
st.title("🎓 Student Apply-Insight Portal")

# -----------------------------
# Data Loading
# -----------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "decisions_enriched.csv"

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

# -----------------------------
# Shared model for prob_yes
# -----------------------------
@st.cache_data(show_spinner=False)
def fit_model_for_probs(df_in: pd.DataFrame, ngrams=(1,2), min_df=3):
    try:
        y = df_in["would_apply"].astype(str).str.lower().eq("yes").astype(int).values
        tfidf = TfidfVectorizer(ngram_range=ngrams, min_df=min_df, stop_words="english")
        X = tfidf.fit_transform(df_in["profile_text"].astype(str).tolist())
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=400)
        clf.fit(X_tr, y_tr)
        acc = float(clf.score(X_te, y_te))
        prob_yes = clf.predict_proba(X)[:, 1]
        return prob_yes, tfidf, clf, acc
    except Exception:
        return None, None, None, None

if "prob_yes" not in df.columns:
    prob_yes, shared_vec, shared_clf, shared_acc = fit_model_for_probs(df, ngrams=(1,2), min_df=3)
    if prob_yes is not None:
        df["prob_yes"] = prob_yes
        df["prob_no"]  = 1 - df["prob_yes"]
else:
    shared_vec = shared_clf = None
    shared_acc = None

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Interactive Controls")
thr = st.sidebar.slider("Decision threshold (P(YES) ≥ …)", 0.0, 1.0, 0.50, 0.01)
use_predictions = st.sidebar.toggle("Use model predictions instead of original Yes/No", value=False)

if "prob_yes" in df.columns:
    df["predicted_apply"] = np.where(df["prob_yes"] >= thr, "yes", "no")
    df["pred_label"] = (df["predicted_apply"] == "yes").astype(int)
else:
    df["predicted_apply"] = df["would_apply"]
    df["pred_label"] = df["label"]

# -----------------------------
# Helpers
# -----------------------------
def _norm_tokens(s: str): return re.findall(r"[a-z0-9\-]+", str(s).lower())

@st.cache_data(show_spinner=False)
def compute_overlap_cols(df_: pd.DataFrame) -> pd.DataFrame:
    df2 = df_.copy()
    def _overlap_row(r):
        p = set(_norm_tokens(r["profile_text"]))
        ra = set(_norm_tokens(r["rationale"]))
        return sorted(p & ra)
    df2["overlap_tokens"] = df2.apply(_overlap_row, axis=1)
    df2["overlap_count"] = df2["overlap_tokens"].apply(len)
    return df2

@st.cache_data(show_spinner=False)
def tfidf_and_drivers(texts, labels, ngrams=(1,2), min_df=3, top_n=30):
    tfidf = TfidfVectorizer(ngram_range=ngrams, min_df=min_df, stop_words="english")
    try:
        X = tfidf.fit_transform(texts)
        terms = np.array(tfidf.get_feature_names_out())
        y = np.asarray(labels)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=400)
        clf.fit(X_tr, y_tr)
        acc = float(clf.score(X_te, y_te))
        coef = clf.coef_.ravel()
        drv_yes = pd.DataFrame({"term": terms[np.argsort(coef)[-top_n:][::-1]], "weight": np.sort(coef)[-top_n:][::-1]})
        drv_no  = pd.DataFrame({"term": terms[np.argsort(coef)[:top_n]], "weight": np.sort(coef)[:top_n]})
        return drv_yes, drv_no, acc, tfidf, clf
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), 0.0, None, None

def summarize_group(df, group_col):
    g = df.groupby(group_col).agg(rows=("label","size"), yes_rate=("pred_label","mean"), avg_prob_yes=("prob_yes","mean")).reset_index()
    g["yes_rate"] = (g["yes_rate"] * 100).round(2)
    g["avg_prob_yes"] = g["avg_prob_yes"].round(3)
    return g.sort_values("avg_prob_yes", ascending=False)

DOMAIN_STOP = {"program","university","degree","course","ms","master","saint","louis","online","format",
                "curriculum","offer","offers","united","states","think","feel","feels","going","like","well",
                "closely","based","right","it","align","aligns","aligned","alignment","fit","fits","fitting",
                "suit","suits","with","my","the","in","for","to","and","or","of","at","on","by","as",
                "analytics","data","background","interests","professional","academic"}

def _has_content_word(phrase, stop):
    for tok in phrase.split():
        t = "".join(ch for ch in tok if ch.isalpha())
        if len(t) >= 4 and t not in stop: return True
    return False

@st.cache_data(show_spinner=False)
def class_distinctive_phrases(df_in, label_col="would_apply", text_col="rationale", ngram_range=(2,3), min_df=3, top_n=15):
    stop = set(ENGLISH_STOP_WORDS) | DOMAIN_STOP
    texts = df_in[text_col].fillna("").astype(str).str.lower().tolist()
    y = df_in[label_col].astype(str).str.lower().tolist()
    vec = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, stop_words=list(stop))
    X = vec.fit_transform(texts)
    terms = vec.get_feature_names_out()
    y_arr = np.array(y)
    yes_mask, no_mask = (y_arr=="yes"), (y_arr=="no")
    yes_mean, no_mean = X[yes_mask].mean(axis=0).A1, X[no_mask].mean(axis=0).A1
    diff = yes_mean - no_mean
    keep = np.array([_has_content_word(t, stop) for t in terms])
    terms_f, diff_f = terms[keep], diff[keep]
    idx_yes, idx_no = diff_f.argsort()[::-1][:top_n], diff_f.argsort()[:top_n]
    top_yes = pd.DataFrame({"phrase": terms_f[idx_yes], "score": diff_f[idx_yes]})
    top_no  = pd.DataFrame({"phrase": terms_f[idx_no], "score": (-diff_f[idx_no])})
    return top_yes, top_no

# -----------------------------
# Derived Stats + Tabs
# -----------------------------
tab_overview, tab_tech = st.tabs(["Non-Technical", "Technical"])

# ======================================================
# 🟢 NON-TECHNICAL TAB
# ======================================================
with tab_overview:
    # KPI tiles
    base_label = "pred_label" if use_predictions else "label"
    yes_pct = 100 * df[base_label].mean()
    no_pct  = 100 - yes_pct
    avg_len = df["rationale"].apply(lambda s: len(str(s).split())).mean()
    c1, c2, c3 = st.columns(3)
    c1.metric(("Predicted Yes %" if use_predictions else "Yes %"), f"{yes_pct:.1f}%")
    c2.metric(("Predicted No %"  if use_predictions else "No %"),  f"{no_pct:.1f}%")
    c3.metric("Avg rationale words", f"{avg_len:.1f}")

    st.markdown("---")

    # Donut + bars
    dec_col = "predicted_apply" if use_predictions else "would_apply"
    yes_count = int((df[dec_col] == "yes").sum())
    no_count  = int((df[dec_col] == "no").sum())
    donut_df = pd.DataFrame({"Decision":["Yes","No"],"Count":[yes_count,no_count]})
    st.plotly_chart(px.pie(donut_df, names="Decision", values="Count", hole=0.55,
                           title=("Predicted Would Apply — Yes vs No" if use_predictions else "Would Apply — Yes vs No")))

    label_for_group = "pred_label" if use_predictions else "label"
    yes_by_bg = df.groupby("academic_background")[label_for_group].mean().mul(100).reset_index().rename(columns={label_for_group:"Yes %"})
    yes_by_exp = df.groupby("previous_work_experience")[label_for_group].mean().mul(100).reset_index().rename(columns={label_for_group:"Yes %"})
    st.plotly_chart(px.bar(yes_by_bg, x="academic_background", y="Yes %", color="Yes %", color_continuous_scale="Blues",
                           title="Likelihood to Apply by Background").update_layout(xaxis_tickangle=-45))
    st.plotly_chart(px.bar(yes_by_exp, x="previous_work_experience", y="Yes %", title="Work Experience Effect"))

    # Distinctive phrases
    st.markdown("---")
    st.subheader("Common rationale keywords (distinctive phrases)")
    col_a, col_b = st.columns(2)
    top_yes_ph, top_no_ph = class_distinctive_phrases(df, label_col=dec_col)
    with col_a:
        st.caption("🟢 Distinctive phrases for YES")
        if not top_yes_ph.empty:
            fig_yes_ph = px.bar(top_yes_ph.iloc[::-1], x="score", y="phrase", orientation="h")
            st.plotly_chart(fig_yes_ph)
    with col_b:
        st.caption("🔴 Distinctive phrases for NO")
        if not top_no_ph.empty:
            fig_no_ph = px.bar(top_no_ph.iloc[::-1], x="score", y="phrase", orientation="h")
            st.plotly_chart(fig_no_ph)

    # Cohort compare
    st.markdown("---")
    st.subheader("Cohort Compare (A/B)")
    left = st.selectbox("Cohort A", sorted(df["academic_background"].dropna().unique()))
    right = st.selectbox("Cohort B", sorted(df["academic_background"].dropna().unique()), index=min(1,len(df["academic_background"].unique())-1))
    A, B = df[df["academic_background"]==left], df[df["academic_background"]==right]
    c1, c2 = st.columns(2)
    for block, name, d in [(c1,left,A),(c2,right,B)]:
        with block:
            st.markdown(f"**{name}**")
            st.metric("Rows", len(d))
            st.metric("Yes %", f"{(100*d[label_for_group].mean() if len(d) else 0):.1f}%")
            if "prob_yes" in d: st.metric("Avg P(YES)", f"{d['prob_yes'].mean():.2f}")
            st.dataframe(d[["academic_background","previous_work_experience",dec_col,"prob_yes","rationale"]].head(15), width="stretch")

    # Search
    st.markdown("---")
    st.subheader("Quick search")
    q = st.text_input("Search text (rationale or profile)", "")
    if q:
        mask = df["rationale"].str.contains(q, case=False, na=False) | df["profile_text"].str.contains(q, case=False, na=False)
        found = df[mask].copy()
        st.caption(f"Matched {len(found):,} rows for “{q}”")
    else:
        found = df.copy()
    show_cols = [c for c in ["academic_background","previous_work_experience",dec_col,"prob_yes","rationale"] if c in found.columns]
    st.dataframe(found.assign(rationale=lambda d: d["rationale"].str.replace(q,f"**{q}**",case=False,regex=False) if q else d["rationale"])[show_cols].head(200), width="stretch")

    # Filter table (color-coded)
    st.markdown("---")
    st.subheader("Filter & Explore")
    df["_background"], df["_experience"], df["_apply"] = df["academic_background"].astype(str), df["previous_work_experience"].astype(str), df["would_apply"].astype(str)
    bg_sel = st.multiselect("Academic background", sorted(df["_background"].unique().tolist()))
    exp_sel = st.multiselect("Previous work experience", sorted(df["_experience"].unique().tolist()))
    apply_sel = st.multiselect("Would apply", ["yes","no"])
    view = df.copy()
    if bg_sel: view = view[view["_background"].isin(bg_sel)]
    if exp_sel: view = view[view["_experience"].isin(exp_sel)]
    if apply_sel: view = view[view["_apply"].isin(apply_sel)]
    display_cols = [c for c in view.columns if c not in ["_background","_experience","_apply"]]
    st.caption(f"Showing {len(view):,} of {len(df):,} rows")
    if "prob_yes" in view:
        styled = view[display_cols].style.format({"prob_yes":"{:.2f}"}).background_gradient(subset=["prob_yes"], cmap="Greens")
        st.dataframe(styled, use_container_width=True)
    else:
        st.dataframe(view[display_cols], width="stretch")

# ======================================================
# 🧪 TECHNICAL TAB — Logistic Regression only + Interactivity
# ======================================================
with tab_tech:
    st.subheader("Deep Dive: Drivers, Overlap & Probabilities")

    # Overlap
    st.markdown("**Profile ↔ Rationale Overlap**")
    df_overlap = compute_overlap_cols(df)
    avg_overlap = df_overlap.groupby("would_apply")["overlap_count"].mean().round(2).to_dict()
    st.write("Average overlap tokens by decision:", avg_overlap)
    st.dataframe(df_overlap[["academic_background","previous_work_experience","would_apply","overlap_tokens","overlap_count"]].head(40), width="stretch")

    st.markdown("---")

    # Logistic Regression Drivers
    st.markdown("### Logistic Regression Drivers")
    st.caption("Trained on TF-IDF features internally; showing learned drivers only.")
    drv_yes, drv_no, acc, vec, clf = tfidf_and_drivers(df["profile_text"].astype(str).tolist(), df["label"].values)
    st.caption(f"Logistic holdout accuracy: **{acc:.3f}**")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Push toward YES**")
        st.dataframe(drv_yes.head(30), width="stretch")
    with c2:
        st.write("**Push toward NO**")
        st.dataframe(drv_no.head(30), width="stretch")
    if not drv_yes.empty:
        fig_drv = px.bar(drv_yes.head(10).iloc[::-1], x="weight", y="term", orientation="h",
                         title="Top-10 YES Drivers")
        st.plotly_chart(fig_drv, use_container_width=True)

    # Probability dist by segment
    st.markdown("---")
    st.subheader("Probability distribution by segment")
    if "prob_yes" in df:
        opt = st.selectbox("Breakdown by…", ["academic_background","previous_work_experience"])
        fig = px.violin(df, x=opt, y="prob_yes", box=True, points="all")
        fig.update_layout(yaxis_title="P(YES)", xaxis_title=opt, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # Per-profile
    st.markdown("---")
    st.subheader("Most confident profiles")
    if "prob_yes" in df:
        c_yes, c_no = st.columns(2)
        with c_yes:
            st.caption("🟢 Highest P(YES)")
            st.dataframe(df.sort_values("prob_yes", ascending=False).head(20)[["academic_background","previous_work_experience","would_apply","prob_yes","rationale"]], width="stretch")
        with c_no:
            st.caption("🔴 Lowest P(YES)")
            st.dataframe(df.sort_values("prob_yes", ascending=True).head(20)[["academic_background","previous_work_experience","would_apply","prob_yes","rationale"]], width="stretch")

st.caption("Data source: data/decisions_enriched.csv • Student Apply-Insight Portal")
