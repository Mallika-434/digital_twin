# C:\Users\malli\OneDrive\Desktop\modules\text_analysis.py

import re
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def _norm_tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9\-]+", str(s).lower())


def _build_profile_text(row: pd.Series) -> str:
    return (
        f"{row.get('academic_background', '')}. "
        f"{row.get('academic_interests', '')}. "
        f"{row.get('professional_interests', '')}. "
        f"gender:{row.get('gender', '')}. "
        f"age:{row.get('age', '')}. "
        f"work:{row.get('previous_work_experience', '')}."
    )


def _sentiment_score(text: str) -> float:
    return TextBlob(str(text)).sentiment.polarity  # [-1, 1]


def _load_program_text(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""


def _overlap_tokens(profile_text: str, rationale: str) -> list[str]:
    p = set(_norm_tokens(profile_text))
    r = set(_norm_tokens(rationale))
    return sorted(p & r)


def run_all(
    decisions_csv: str | Path,
    program_txt: str | Path,
    *,
    min_df: int = 3,
    ngrams: tuple[int, int] = (1, 2),
    top_n: int = 30,
) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Returns:
      stats: dict of headline KPIs
      tfidf_out: {'top_yes_terms': df, 'top_no_terms': df}
      drivers_out: {'drivers_yes': df, 'drivers_no': df, 'holdout_accuracy': float}
      df_enriched: original df with extra columns (overlap, program refs, sentiment)
    """
    decisions_csv = Path(decisions_csv)
    program_txt = Path(program_txt)

    df = pd.read_csv(decisions_csv)

    # Required columns sanity
    required_cols = [
        "gender","age","academic_background","race",
        "academic_interests","professional_interests","previous_work_experience",
        "would_apply","rationale","auto_fixed"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in decisions CSV: {missing_cols}")

    # Normalize fields
    df["would_apply"] = df["would_apply"].astype(str).str.lower().str.strip()
    df = df[df["would_apply"].isin(["yes","no"])].copy()
    df["label"] = (df["would_apply"] == "yes").astype(int)
    df["rationale"] = df["rationale"].fillna("").astype(str)

    # Derived text
    df["profile_text"] = df.apply(_build_profile_text, axis=1)

    # KPIs
    yes_pct = 100 * df["label"].mean()
    no_pct = 100 - yes_pct
    avg_rationale_len = df["rationale"].apply(lambda s: len(s.split())).mean()
    auto_fixed_pct = 100 * df["auto_fixed"].astype(str).str.lower().eq("yes").mean()
    yes_by_bg = (df.groupby("academic_background")["label"].mean().mul(100).sort_values(ascending=False))
    yes_by_exp = (df.groupby("previous_work_experience")["label"].mean().mul(100).sort_index())

    stats = {
        "yes_pct": round(yes_pct, 1),
        "no_pct": round(no_pct, 1),
        "avg_rationale_words": round(avg_rationale_len, 1),
        "auto_fixed_pct": round(auto_fixed_pct, 1),
        "yes_by_background": yes_by_bg,
        "yes_by_experience": yes_by_exp,
        "rows": len(df),
    }

    # Overlap tokens
    df["overlap_tokens"] = df.apply(lambda r: _overlap_tokens(r["profile_text"], r["rationale"]), axis=1)
    df["overlap_count"] = df["overlap_tokens"].apply(len)

    # TF-IDF class signatures
    tfidf = TfidfVectorizer(ngram_range=ngrams, min_df=min_df, stop_words="english")
    X = tfidf.fit_transform(df["profile_text"])
    y = df["label"].values
    terms = np.array(tfidf.get_feature_names_out())

    yes_mean = X[y == 1].mean(axis=0)
    no_mean = X[y == 0].mean(axis=0)
    diff = np.asarray(yes_mean - no_mean).ravel()

    top_yes_idx = diff.argsort()[::-1][:top_n]
    top_no_idx = diff.argsort()[:top_n]

    top_yes_terms = pd.DataFrame({"term": terms[top_yes_idx], "score": diff[top_yes_idx]})
    top_no_terms = pd.DataFrame({"term": terms[top_no_idx], "score": -diff[top_no_idx]})
    tfidf_out = {"top_yes_terms": top_yes_terms, "top_no_terms": top_no_terms}

    # Logistic Regression drivers (supervised)
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
    drivers_out = {"drivers_yes": drivers_yes, "drivers_no": drivers_no, "holdout_accuracy": acc}

    # Program-text influence
    program_text = _load_program_text(program_txt)
    prog_tokens = set(_norm_tokens(program_text))
    df["rationale_tokens"] = df["rationale"].apply(_norm_tokens)
    df["program_refs_in_rationale"] = df["rationale_tokens"].apply(
        lambda toks: sum(1 for t in toks if t in prog_tokens)
    )

    # Sentiment
    df["rationale_sentiment"] = df["rationale"].apply(_sentiment_score)

    # Enriched DF subset
    export_cols = [
        "gender","age","academic_background","race",
        "academic_interests","professional_interests","previous_work_experience",
        "would_apply","rationale","auto_fixed",
        "profile_text","overlap_tokens","overlap_count",
        "program_refs_in_rationale","rationale_sentiment"
    ]
    df_enriched = df[export_cols].copy()

    return stats, tfidf_out, drivers_out, df_enriched
