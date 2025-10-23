# app.py — Student Apply-Insight Portal (Final Version with Decision Flow Diagram + Filter Chart + Tooltips)

from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import graphviz
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
# 3️⃣ Model fitting for probabilities
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
# 4️⃣ Sidebar Controls
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

# -------------------------------------------------
# 6️⃣ Tabs
# -------------------------------------------------
tab_overview, tab_tech = st.tabs(["Non-Technical", "Technical"])

# =================================================
# 🟢 NON-TECHNICAL TAB
# =================================================
with tab_overview:
    base_label = "pred_label" if use_predictions else "label"
    yes_pct = 100 * df[base_label].mean()
    no_pct = 100 - yes_pct
    avg_len = df["rationale"].apply(lambda s: len(str(s).split())).mean()

    c1, c2, c3 = st.columns(3)
    c1.metric(("Predicted Yes %" if use_predictions else "Yes %"), f"{yes_pct:.1f}%")
    c2.metric(("Predicted No %" if use_predictions else "No %"), f"{no_pct:.1f}%")
    c3.metric("Avg rationale words", f"{avg_len:.1f}")
    st.markdown("---")

    with st.container():
        st.subheader("📊 Decision Insights")
        dec_col = "predicted_apply" if use_predictions else "would_apply"
        yes_count = int((df[dec_col] == "yes").sum())
        no_count = int((df[dec_col] == "no").sum())
        donut_df = pd.DataFrame({"Decision": ["Yes", "No"], "Count": [yes_count, no_count]})
        st.plotly_chart(px.pie(donut_df, names="Decision", values="Count", hole=0.55), use_container_width=True)

        yes_by_bg = df.groupby("academic_background")[base_label].mean().mul(100).reset_index()
        yes_by_bg.rename(columns={base_label: "Yes %"}, inplace=True)
        st.plotly_chart(px.bar(yes_by_bg, x="academic_background", y="Yes %", color="Yes %",
                               color_continuous_scale="Blues", title="Yes % by Academic Background").update_layout(xaxis_tickangle=-45),
                        use_container_width=True)

        yes_by_exp = df.groupby("previous_work_experience")[base_label].mean().mul(100).reset_index()
        yes_by_exp.rename(columns={base_label: "Yes %"}, inplace=True)
        st.plotly_chart(px.bar(yes_by_exp, x="previous_work_experience", y="Yes %", title="Work Experience Effect"),
                        use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Filter & Explore")

    df["_background"], df["_experience"], df["_apply"] = (
        df["academic_background"].astype(str),
        df["previous_work_experience"].astype(str),
        df["would_apply"].astype(str)
    )

    bg_sel = st.multiselect("Academic background", sorted(df["_background"].unique()), help="Filter data by academic background.")
    exp_sel = st.multiselect("Previous work experience", sorted(df["_experience"].unique()), help="Filter data by experience.")
    apply_sel = st.multiselect("Would apply", ["yes", "no"], help="Filter data by decision outcome.")

    view = df.copy()
    if bg_sel: view = view[view["_background"].isin(bg_sel)]
    if exp_sel: view = view[view["_experience"].isin(exp_sel)]
    if apply_sel: view = view[view["_apply"].isin(apply_sel)]

    st.caption(f"Showing {len(view):,} of {len(df):,} rows")

    # --- NEW: Filtered Yes/No Graph ---
    if len(view) > 0:
        yes_count = (view["would_apply"].str.lower() == "yes").sum()
        no_count = (view["would_apply"].str.lower() == "no").sum()
        pct_yes = (yes_count / len(view)) * 100
        pct_no = (no_count / len(view)) * 100
        chart_df = pd.DataFrame({"Decision": ["Yes", "No"], "Percentage": [pct_yes, pct_no]})
        st.plotly_chart(px.bar(chart_df, x="Decision", y="Percentage", color="Decision",
                               color_discrete_map={"Yes": "green", "No": "red"},
                               text_auto=".1f", title="Filtered Yes/No Percentage").update_layout(yaxis_title="%", showlegend=False),
                        use_container_width=True)
    else:
        st.info("No matching records for the selected filters.")

    display_cols = [c for c in view.columns if c not in ["_background", "_experience", "_apply"]]
    styled = view[display_cols].style.format({"prob_yes": "{:.2f}"}).background_gradient(subset=["prob_yes"], cmap="Greens")
    st.dataframe(styled, use_container_width=True)

# =================================================
# 🧪 TECHNICAL TAB
# =================================================
with tab_tech:
    st.subheader("Deep Dive: Drivers, Overlap & Probabilities")

    # --- NEW: Decision Flow Diagram ---
    st.markdown("### 🧭 How the Decision is Made")
    st.caption("This flow shows how each profile is classified as 'YES' or 'NO' depending on whether we use the LLM or the logistic regression model.")
    decision_flow = graphviz.Digraph()
    decision_flow.attr(rankdir="LR", size="6,2")
    decision_flow.node("A", "Start (Profile Input)", shape="ellipse", style="filled", color="#ADD8E6")
    decision_flow.node("B", "Mode Selected?", shape="diamond", style="filled", color="#FFD700")
    decision_flow.node("C1", "LLM Mode → Phi-3 (Ollama)", shape="box", style="filled", color="#C6EFCE")
    decision_flow.node("C2", "Model Mode → Logistic Regression", shape="box", style="filled", color="#C6EFCE")
    decision_flow.node("D1", "LLM Output → would_apply ('yes'/'no')", shape="ellipse", color="#A9D08E")
    decision_flow.node("D2", "TF-IDF Features → P(YES)", shape="box", color="#A9D08E")
    decision_flow.node("E", "Compare with Threshold → Predict 'YES' if ≥ thr", shape="ellipse", color="#A9D08E")
    decision_flow.node("F", "Final Decision (YES / NO)", shape="ellipse", style="filled", color="#FFA07A")
    decision_flow.edge("A", "B")
    decision_flow.edge("B", "C1", label="LLM Mode")
    decision_flow.edge("B", "C2", label="Model Mode")
    decision_flow.edge("C1", "D1")
    decision_flow.edge("C2", "D2")
    decision_flow.edge("D2", "E")
    decision_flow.edge("D1", "F")
    decision_flow.edge("E", "F")
    st.graphviz_chart(decision_flow)
    st.markdown("---")

    # --- Overlap & Drivers ---
    df_overlap = compute_overlap_cols(df)
    avg_overlap = df_overlap.groupby("would_apply")["overlap_count"].mean().round(2).to_dict()
    st.write("Average overlap tokens by decision:", avg_overlap)
    st.dataframe(df_overlap[["academic_background", "previous_work_experience", "would_apply", "overlap_tokens", "overlap_count"]].head(40), use_container_width=True)

    st.markdown("---")
    st.markdown("### Logistic Regression Drivers (n-grams = (1, 2))")
    top_n = st.number_input("Top-N terms", min_value=10, max_value=50, value=30, step=1, help="Enter how many top weighted terms you want to display (10–50).")
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
        fig_drv = px.bar(drv_yes.head(10).iloc[::-1], x="weight", y="term", orientation="h", title="Top-10 YES Drivers")
        st.plotly_chart(fig_drv, use_container_width=True)

    st.markdown("---")
    st.markdown("### Model Probability (per profile)")
    if vec is not None and clf is not None:
        X_all = vec.transform(df["profile_text"])
        df["prob_yes"] = clf.predict_proba(X_all)[:, 1]
        df["prob_no"] = 1 - df["prob_yes"]
        fig_prob = px.histogram(df, x="prob_yes", nbins=20, title="Model Confidence: P(YES)")
        fig_prob.update_layout(xaxis_title="P(YES)", yaxis_title="Count")
        st.plotly_chart(fig_prob, use_container_width=True)

st.caption("Data source: data/decisions_enriched.csv • Student Apply-Insight Portal")
