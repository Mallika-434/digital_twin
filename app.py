import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Apply-Insight Portal", layout="wide")
st.title("Student Apply-Insight Portal")

from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Apply-Insight Portal", layout="wide")

# ✅ Always load from the repo's /data folder (works locally + Streamlit Cloud)
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
    df = df[df["would_apply"].isin(["yes","no"])].copy()
    df["label"] = (df["would_apply"] == "yes").astype(int)
    df["rationale"] = df["rationale"].fillna("").astype(str)
    return df

df = load_df(CSV_PATH)


# Quick derived cols
df["label"] = (df["would_apply"].str.lower() == "yes").astype(int)
yes_pct = 100 * df["label"].mean()
no_pct = 100 - yes_pct
avg_len = df["rationale"].fillna("").apply(lambda s: len(str(s).split())).mean()
autofixed_pct = 100 * df["auto_fixed"].astype(str).str.lower().eq("yes").mean()

tab_overview, tab_tech = st.tabs(["Non-Technical", "Technical"])

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Yes %", f"{yes_pct:.1f}%")
    c2.metric("No %", f"{no_pct:.1f}%")
    c3.metric("Avg rationale words", f"{avg_len:.1f}")
    c4.metric("% auto-fixed", f"{autofixed_pct:.1f}%")

    st.subheader("Yes % by Academic Background")
    st.bar_chart(df.groupby("academic_background")["label"].mean().sort_values().mul(100))

    st.subheader("Table (filterable)")
    bg = st.multiselect("Filter by background", sorted(df["academic_background"].astype(str).unique().tolist()))
    view = df if not bg else df[df["academic_background"].isin(bg)]
    st.dataframe(view, width="stretch")

with tab_tech:
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(50), width="stretch")
    st.caption("We’ll add TF-IDF, logistic drivers, overlaps, and program influence here next.")

st.caption("Local demo using decisions_enriched.csv")
