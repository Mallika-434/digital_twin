import pandas as pd
import streamlit as st

st.set_page_config(page_title="Apply-Insight Portal", layout="wide")
st.title("🎓 Apply-Insight Portal")

# Load data
CSV_PATH = r"C:\Users\malli\OneDrive\Desktop\outputs\decisions_enriched.csv"
df = pd.read_csv(CSV_PATH)

# Quick derived cols
df["label"] = (df["would_apply"].str.lower() == "yes").astype(int)
yes_pct = 100 * df["label"].mean()
no_pct = 100 - yes_pct
avg_len = df["rationale"].fillna("").apply(lambda s: len(str(s).split())).mean()
autofixed_pct = 100 * df["auto_fixed"].astype(str).str.lower().eq("yes").mean()

tab_overview, tab_tech = st.tabs(["Overview (Non-Technical)", "Lab (Technical)"])

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
    st.dataframe(view, use_container_width=True)

with tab_tech:
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(50), use_container_width=True)
    st.caption("We’ll add TF-IDF, logistic drivers, overlaps, and program influence here next.")

st.caption("Local demo using decisions_enriched.csv")
