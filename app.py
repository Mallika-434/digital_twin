# app.py — Student Apply-Insight Portal

from pathlib import Path
import pandas as pd
import streamlit as st

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
    """Safely load and clean the CSV data."""
    if not path.exists():
        st.error(f"❌ File not found: {path}")
        st.write("Available files in /data:", [p.name for p in DATA_DIR.glob('*')])
        st.stop()
    df = pd.read_csv(path)
    df["would_apply"] = df["would_apply"].astype(str).str.lower().str.strip()
    df = df[df["would_apply"].isin(["yes", "no"])].copy()
    df["label"] = (df["would_apply"] == "yes").astype(int)
    df["rationale"] = df["rationale"].fillna("").astype(str)
    return df

df = load_df(CSV_PATH)

# -----------------------------
# Derived Stats
# -----------------------------
df["label"] = (df["would_apply"].str.lower() == "yes").astype(int)
yes_pct = 100 * df["label"].mean()
no_pct = 100 - yes_pct
avg_len = df["rationale"].fillna("").apply(lambda s: len(str(s).split())).mean()
autofixed_pct = 100 * df["auto_fixed"].astype(str).str.lower().eq("yes").mean()

# -----------------------------
# Tabs setup
# -----------------------------
tab_overview, tab_tech = st.tabs(["Non-Technical", "Technical"])

# ======================================================
# 🟢 NON-TECHNICAL TAB
# ======================================================
with tab_overview:
    # KPI tiles
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Yes %", f"{yes_pct:.1f}%")
    c2.metric("No %", f"{no_pct:.1f}%")
    c3.metric("Avg rationale words", f"{avg_len:.1f}")
    c4.metric("% auto-fixed", f"{autofixed_pct:.1f}%")

    st.markdown("---")

    # --- Filter & Explore (three-field filter) ---
    st.subheader("Filter & Explore")

    # Ensure clean string columns for filters
    df["_background"] = df["academic_background"].astype(str)
    df["_experience"] = df["previous_work_experience"].astype(str)
    df["_apply"] = df["would_apply"].astype(str).str.lower()

    colf1, colf2, colf3, colf4 = st.columns([1, 1, 1, 0.5])

    with colf1:
        bg_sel = st.multiselect(
            "Academic background",
            options=sorted(df["_background"].unique().tolist()),
            default=[],
            help="Select one or more backgrounds; leave empty for all."
        )

    with colf2:
        exp_sel = st.multiselect(
            "Previous work experience",
            options=sorted(df["_experience"].unique().tolist()),
            default=[],
            help="Select Yes/No (or leave empty for all)."
        )

    with colf3:
        apply_sel = st.multiselect(
            "Would apply",
            options=["yes", "no"],
            default=[],
            help="Filter by decision; leave empty for all."
        )

    with colf4:
        reset = st.button("Reset filters")

    # Apply filters
    view = df.copy()
    if reset:
        bg_sel, exp_sel, apply_sel = [], [], []

    if bg_sel:
        view = view[view["_background"].isin(bg_sel)]
    if exp_sel:
        view = view[view["_experience"].isin(exp_sel)]
    if apply_sel:
        view = view[view["_apply"].isin([a.lower() for a in apply_sel])]

    # Drop helper cols from display
    display_cols = [c for c in view.columns if c not in ["_background", "_experience", "_apply"]]

    # Summary + table
    st.caption(f"Showing {len(view):,} of {len(df):,} rows")
    st.dataframe(view[display_cols], width="stretch")

    # Download exactly what’s shown
    st.download_button(
        label="Download filtered CSV",
        data=view[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="decisions_filtered.csv",
        mime="text/csv",
    )

# ======================================================
# 🧪 TECHNICAL TAB
# ======================================================
with tab_tech:
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(50), width="stretch")
    st.caption("We'll add TF-IDF, logistic drivers, overlaps, and program influence here next.")

# Footer
st.caption("Data source: data/decisions_enriched.csv • Student Apply-Insight Portal")
