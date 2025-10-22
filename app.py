# app.py — Student Apply-Insight Portal (no auto-fixed; phrase analytics)

from pathlib import Path
import re
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

# ML
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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

    # Auto-drop auto_fixed if present
    if "auto_fixed" in df.columns:
        df.drop(columns=["auto_fixed"], inplace=True)

    # Build profile_text if missing
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
# Helpers (TECH tab + phrases)
# -----------------------------
def _norm_tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9\-]+", str(s).lower())

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
def tfidf_and_drivers(
    texts: list[str],
    labels: np.ndarray,
    ngrams: Tuple[int, int] = (1, 2),
    min_df: int = 3,
    top_n: int = 30,
):
    tfidf = TfidfVectorizer(ngram_range=ngrams, min_df=min_df, stop_words="english")
    try:
        X = tfidf.fit_transform(texts)
    except ValueError:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0.0, None, None)

    terms = np.array(tfidf.get_feature_names_out())
    y = np.asarray(labels)

    yes_mean = X[y == 1].mean(axis=0)
    no_mean  = X[y == 0].mean(axis=0)
    diff = np.asarray(yes_mean - no_mean).ravel()

    top_yes_idx = diff.argsort()[::-1][:top_n]
    top_no_idx  = diff.argsort()[:top_n]

    top_yes_terms = pd.DataFrame({"term": terms[top_yes_idx], "score": diff[top_yes_idx]})
    top_no_terms  = pd.DataFrame({"term": terms[top_no_idx],  "score": -diff[top_no_idx]})

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=400)
        clf.fit(X_tr, y_tr)
        acc = float(clf.score(X_te, y_te))
        coef = clf.coef_.ravel()

        drv_yes_idx = np.argsort(coef)[-top_n:][::-1]
        drv_no_idx  = np.argsort(coef)[:top_n]

        drivers_yes = pd.DataFrame({"term": terms[drv_yes_idx], "weight": coef[drv_yes_idx]})
        drivers_no  = pd.DataFrame({"term": terms[drv_no_idx],  "weight": coef[drv_no_idx]})
    except Exception:
        acc = 0.0
        clf = None
        drivers_yes = pd.DataFrame()
        drivers_no  = pd.DataFrame()

    return top_yes_terms, top_no_terms, drivers_yes, drivers_no, acc, tfidf, clf

def summarize_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    g = (
        df.groupby(group_col)
          .agg(
              rows=("label","size"),
              yes_rate=("label","mean"),
              avg_prob_yes=("prob_yes","mean")
          )
          .reset_index()
    )
    g["yes_rate"] = (g["yes_rate"] * 100).round(2)
    g["avg_prob_yes"] = g["avg_prob_yes"].round(3)
    return g.sort_values("avg_prob_yes", ascending=False)

# ---------- Rationale analytics helpers ----------
DOMAIN_STOP = {
    # generic program words
    "program","university","degree","course","ms","master","saint","louis",
    "united","states","online","format","curriculum","offer","offers",
    # filler
    "think","feel","feels","going","like","well","closely","based",
}
def _prep_rationale(s: pd.Series) -> list[str]:
    return s.fillna("").astype(str).str.lower().tolist()

def top_ngrams_by_class(df_in: pd.DataFrame, label_col: str, text_col: str,
                        y_value: str, ngram_range=(2,3), top_n=15,
                        extra_stop=set()) -> pd.DataFrame:
    texts = _prep_rationale(df_in.loc[df_in[label_col] == y_value, text_col])
    if len(texts) == 0:
        return pd.DataFrame(columns=["phrase","count"])
    stop = set(extra_stop) | DOMAIN_STOP
    vec = CountVectorizer(ngram_range=ngram_range, stop_words=list(stop), min_df=1)
    X = vec.fit_transform(texts)
    counts = X.sum(axis=0).A1
    phrases = np.array(vec.get_feature_names_out())
    order = counts.argsort()[::-1][:top_n]
    return pd.DataFrame({"phrase": phrases[order], "count": counts[order]})

def highlight_phrases(text: str, phrases: list[str]) -> str:
    import re as _re
    t = text
    for p in sorted(phrases, key=len, reverse=True)[:10]:
        pattern = _re.compile(_re.escape(p), _re.IGNORECASE)
        t = pattern.sub(f"<mark>{p}</mark>", t)
    return t

def tag_rationale(s: str) -> list[str]:
    s = (s or "").lower()
    tags = []
    if any(k in s for k in ["fit","align","match","suits","relevant"]): tags.append("fit")
    if any(k in s for k in ["not fit","mismatch","different field","unrelated"]): tags.append("mismatch")
    if any(k in s for k in ["deadline","time","october","soon","close"]): tags.append("deadline")
    if any(k in s for k in ["experience","worked","intern","job"]): tags.append("experience")
    if any(k in s for k in ["skills","learn","deepen","build skills"]): tags.append("skills")
    return tags

# -----------------------------
# Derived Stats (Non-Tech KPIs)
# -----------------------------
df["label"] = (df["would_apply"].str.lower() == "yes").astype(int)
yes_pct = 100 * df["label"].mean()
no_pct = 100 - yes_pct
avg_len = df["rationale"].apply(lambda s: len(str(s).split())).mean()

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_tech = st.tabs(["Non-Technical", "Technical"])

# ======================================================
# 🟢 NON-TECHNICAL TAB
# ======================================================
with tab_overview:
    # KPI tiles (no auto-fixed)
    c1, c2, c3 = st.columns(3)
    c1.metric("Yes %", f"{yes_pct:.1f}%")
    c2.metric("No %", f"{no_pct:.1f}%")
    c3.metric("Avg rationale words", f"{avg_len:.1f}")

    st.markdown("---")

    # Overall outcome
    st.subheader("Overall outcome")
    yes_count = int((df["would_apply"] == "yes").sum())
    no_count  = int((df["would_apply"] == "no").sum())
    donut_df = pd.DataFrame({"Decision": ["Yes", "No"], "Count": [yes_count, no_count]})
    st.plotly_chart(px.pie(donut_df, names="Decision", values="Count", hole=0.55,
                           title="Would Apply — Yes vs No"))

    # Yes% by background
    st.subheader("Yes % by Academic Background", anchor="yes-by-academic-background")
    yes_by_bg = (
        df.groupby("academic_background")["label"].mean().mul(100).reset_index().rename(columns={"label": "Yes %"})
    )
    yes_by_bg = yes_by_bg.sort_values("Yes %", ascending=False)
    st.plotly_chart(px.bar(yes_by_bg, x="academic_background", y="Yes %",
                           color="Yes %", color_continuous_scale="Blues",
                           title="Likelihood to Apply by Background").update_layout(
                               xaxis_title="Academic Background", yaxis_title="Yes %",
                               xaxis_tickangle=-45, showlegend=False))

    # Yes% by experience
    st.subheader("Yes % by Work Experience")
    yes_by_exp = (
        df.groupby("previous_work_experience")["label"].mean().mul(100).reset_index()
        .rename(columns={"label": "Yes %", "previous_work_experience": "Work Experience"})
    )
    st.plotly_chart(px.bar(yes_by_exp, x="Work Experience", y="Yes %",
                           title="Work Experience Effect"))

    st.markdown("---")

    # 🔁 NEW: Phrase analytics instead of wordclouds
    st.subheader("Common rationale keywords (phrases)")
    col_a, col_b = st.columns(2)

    top_yes_ph = top_ngrams_by_class(df, "would_apply", "rationale", "yes",
                                     ngram_range=(2,3), top_n=15, extra_stop=set())
    top_no_ph  = top_ngrams_by_class(df, "would_apply", "rationale", "no",
                                     ngram_range=(2,3), top_n=15, extra_stop=set())

    with col_a:
        st.caption("🟢 Frequent phrases in YES rationales")
        if not top_yes_ph.empty:
            fig_yes_ph = px.bar(top_yes_ph.iloc[::-1], x="count", y="phrase",
                                orientation="h", title="Top phrases (YES)")
            fig_yes_ph.update_layout(yaxis_title="", xaxis_title="Count")
            st.plotly_chart(fig_yes_ph)
            st.download_button(
                "Download YES phrases (CSV)",
                data=top_yes_ph.to_csv(index=False).encode("utf-8"),
                file_name="yes_rationale_phrases.csv",
                mime="text/csv",
            )
        else:
            st.info("No YES rationales available.")

    with col_b:
        st.caption("🔴 Frequent phrases in NO rationales")
        if not top_no_ph.empty:
            fig_no_ph = px.bar(top_no_ph.iloc[::-1], x="count", y="phrase",
                               orientation="h", title="Top phrases (NO)")
            fig_no_ph.update_layout(yaxis_title="", xaxis_title="Count")
            st.plotly_chart(fig_no_ph)
            st.download_button(
                "Download NO phrases (CSV)",
                data=top_no_ph.to_csv(index=False).encode("utf-8"),
                file_name="no_rationale_phrases.csv",
                mime="text/csv",
            )
        else:
            st.info("No NO rationales available.")

    st.markdown("---")

    # Representative rationales with highlights + simple theme tags
    st.subheader("Representative rationales (with highlights)")
    yes_phr_list = top_yes_ph["phrase"].head(10).tolist() if not top_yes_ph.empty else []
    no_phr_list  = top_no_ph["phrase"].head(10).tolist() if not top_no_ph.empty else []

    c_yes, c_no = st.columns(2)

    with c_yes:
        st.caption("🟢 YES examples")
        view_yes = df[df["would_apply"]=="yes"].copy()
        if "prob_yes" in df.columns:
            view_yes = view_yes.sort_values("prob_yes", ascending=False).head(5)
        else:
            view_yes = view_yes.head(5)
        for _, r in view_yes.iterrows():
            html = highlight_phrases(str(r["rationale"]), yes_phr_list)
            tags = ", ".join(tag_rationale(r["rationale"]))
            st.markdown(
                f"- **Background:** {r.get('academic_background','—')}  \n"
                f"  *{r.get('previous_work_experience','—')}*  \n"
                + (f"  **P(YES): {r['prob_yes']:.2f}**  \n" if 'prob_yes' in r else "")
                + (f"  _tags: {tags}_  \n" if tags else "")
                + f"  <div style='margin-top:4px'>{html}</div>",
                unsafe_allow_html=True
            )

    with c_no:
        st.caption("🔴 NO examples")
        view_no = df[df["would_apply"]=="no"].copy()
        if "prob_yes" in df.columns:
            view_no = view_no.sort_values("prob_yes", ascending=True).head(5)
        else:
            view_no = view_no.head(5)
        for _, r in view_no.iterrows():
            html = highlight_phrases(str(r["rationale"]), no_phr_list)
            tags = ", ".join(tag_rationale(r["rationale"]))
            st.markdown(
                f"- **Background:** {r.get('academic_background','—')}  \n"
                f"  *{r.get('previous_work_experience','—')}*  \n"
                + (f"  **P(YES): {r['prob_yes']:.2f}**  \n" if 'prob_yes' in r else "")
                + (f"  _tags: {tags}_  \n" if tags else "")
                + f"  <div style='margin-top:4px'>{html}</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # Filterable table (3-field filter)
    st.subheader("Filter & Explore")
    df["_background"] = df["academic_background"].astype(str)
    df["_experience"] = df["previous_work_experience"].astype(str)
    df["_apply"] = df["would_apply"].astype(str).str.lower()

    colf1, colf2, colf3, colf4 = st.columns([1, 1, 1, 0.5])
    with colf1:
        bg_sel = st.multiselect("Academic background", sorted(df["_background"].unique().tolist()))
    with colf2:
        exp_sel = st.multiselect("Previous work experience", sorted(df["_experience"].unique().tolist()))
    with colf3:
        apply_sel = st.multiselect("Would apply", ["yes", "no"])
    with colf4:
        reset = st.button("Reset filters")

    view = df.copy()
    if reset:
        bg_sel, exp_sel, apply_sel = [], [], []
    if bg_sel:
        view = view[view["_background"].isin(bg_sel)]
    if exp_sel:
        view = view[view["_experience"].isin(exp_sel)]
    if apply_sel:
        view = view[view["_apply"].isin([a.lower() for a in apply_sel])]

    display_cols = [c for c in view.columns if c not in ["_background", "_experience", "_apply"]]
    st.caption(f"Showing {len(view):,} of {len(df):,} rows")
    st.dataframe(view[display_cols], width="stretch")
    st.download_button(
        "Download filtered CSV",
        data=view[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="decisions_filtered.csv",
        mime="text/csv",
    )

# ======================================================
# 🧪 TECHNICAL TAB
# ======================================================
with tab_tech:
    st.subheader("Deep Dive: Tokens, Drivers, Overlap & Probabilities")

    # Overlap
    st.markdown("**Profile ↔ Rationale Overlap**")
    df_overlap = compute_overlap_cols(df)
    avg_overlap = df_overlap.groupby("would_apply")["overlap_count"].mean().round(2).to_dict()
    st.write("Average overlap tokens by decision:", avg_overlap)
    st.dataframe(
        df_overlap[["academic_background","previous_work_experience","would_apply","overlap_tokens","overlap_count"]]
        .sort_values("overlap_count", ascending=False)
        .head(40),
        width="stretch"
    )
    st.download_button(
        "Download overlap sample (CSV)",
        data=df_overlap.nlargest(200, "overlap_count")[["academic_background","previous_work_experience","would_apply","overlap_tokens","overlap_count"]]
            .to_csv(index=False).encode("utf-8"),
        file_name="overlap_top200.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # Keyword Signals
    st.markdown("### Keyword Signals")
    st.caption("TF-IDF class signatures (Yes/No) + Logistic Regression drivers.")

    colcfg1, colcfg2 = st.columns([1, 1])
    ngram_opt = colcfg1.selectbox("n-grams", ["1", "1-2"], index=1)
    ngrams = (1, 2) if ngram_opt == "1-2" else (1, 1)
    top_n = colcfg2.slider("Top-N terms", 10, 50, 30, 5)

    FIXED_MIN_DF = 3
    top_yes, top_no, drv_yes, drv_no, acc, vec, clf = tfidf_and_drivers(
        df["profile_text"].astype(str).tolist(),
        df["label"].values,
        ngrams=ngrams, min_df=FIXED_MIN_DF, top_n=top_n
    )
    st.caption(f"Logistic holdout accuracy (sanity check): **{acc:.3f}**")

    st.markdown("##### Class Signatures (TF-IDF difference)")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Characteristic of YES**")
        st.dataframe(top_yes.head(top_n), width="stretch")
        st.download_button("Download TF-IDF YES terms",
                           data=top_yes.to_csv(index=False).encode("utf-8"),
                           file_name="tfidf_yes_terms.csv", mime="text/csv")
    with c2:
        st.write("**Characteristic of NO**")
        st.dataframe(top_no.head(top_n), width="stretch")
        st.download_button("Download TF-IDF NO terms",
                           data=top_no.to_csv(index=False).encode("utf-8"),
                           file_name="tfidf_no_terms.csv", mime="text/csv")

    st.markdown("---")

    st.markdown("##### Directional Drivers (Logistic Regression coefficients)")
    c3, c4 = st.columns(2)
    with c3:
        st.write("**Push toward YES**")
        st.dataframe(drv_yes.head(top_n), width="stretch")
        st.download_button("Download YES drivers",
                           data=drv_yes.to_csv(index=False).encode("utf-8"),
                           file_name="drivers_yes.csv", mime="text/csv")
    with c4:
        st.write("**Push toward NO**")
        st.dataframe(drv_no.head(top_n), width="stretch")
        st.download_button("Download NO drivers",
                           data=drv_no.to_csv(index=False).encode("utf-8"),
                           file_name="drivers_no.csv", mime="text/csv")

    st.markdown("##### Top-10 YES Drivers (visual)")
    if not drv_yes.empty:
        top10_yes = drv_yes.head(10).iloc[::-1]
        fig_drv = px.bar(top10_yes, x="weight", y="term", orientation="h",
                         title="Terms that push the model toward a YES decision")
        fig_drv.update_layout(xaxis_title="Coefficient weight (↑ = more YES)", yaxis_title="Term")
        st.plotly_chart(fig_drv)
    else:
        st.info("Drivers not available with current settings or dataset size.")

    st.markdown("---")

    # Model probabilities per profile
    st.markdown("### Model Probability (per profile)")
    if vec is not None and clf is not None:
        X_all = vec.transform(df["profile_text"].astype(str).tolist())
        df["prob_yes"] = clf.predict_proba(X_all)[:, 1]
        df["prob_no"] = 1 - df["prob_yes"]

        st.subheader("Distribution of YES probabilities")
        fig_prob = px.histogram(df, x="prob_yes", nbins=20, title="Model confidence: P(YES)")
        fig_prob.update_layout(xaxis_title="P(YES)", yaxis_title="Count")
        st.plotly_chart(fig_prob)

        st.subheader("Most confident profiles")
        c_yes, c_no = st.columns(2)
        with c_yes:
            st.caption("🟢 Highest P(YES)")
            st.dataframe(
                df.sort_values("prob_yes", ascending=False)
                  .head(20)[["academic_background","previous_work_experience","would_apply","prob_yes","rationale"]],
                width="stretch"
            )
        with c_no:
            st.caption("🔴 Lowest P(YES)")
            st.dataframe(
                df.sort_values("prob_yes", ascending=True)
                  .head(20)[["academic_background","previous_work_experience","would_apply","prob_yes","rationale"]],
                width="stretch"
            )

        st.markdown("---")
        st.subheader("Average model probability by group")

        cols = st.columns(2)
        with cols[0]:
            if "academic_background" in df.columns:
                g_bg = summarize_group(df, "academic_background")
                st.write("**By Academic Background**")
                st.dataframe(g_bg, width="stretch")
                st.download_button("Download summary (background)",
                                   data=g_bg.to_csv(index=False).encode("utf-8"),
                                   file_name="avg_prob_by_background.csv", mime="text/csv")
        with cols[1]:
            if "previous_work_experience" in df.columns:
                g_exp = summarize_group(df, "previous_work_experience")
                st.write("**By Work Experience**")
                st.dataframe(g_exp, width="stretch")
                st.download_button("Download summary (experience)",
                                   data=g_exp.to_csv(index=False).encode("utf-8"),
                                   file_name="avg_prob_by_experience.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Download enriched data")
        st.download_button(
            "Download decisions with probabilities (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="decisions_with_probs.csv",
            mime="text/csv",
        )
    else:
        st.info("Model not available for probability scoring with current settings.")

# Footer
st.caption("Data source: data/decisions_enriched.csv • Student Apply-Insight Portal")
