"""
CleanSense AI — Data Quality Scorer
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from services.detector import detect_missing_values, detect_duplicates, detect_outliers


def calculate_data_quality_score(df: pd.DataFrame) -> Tuple[int, Dict[str, float]]:
    """
    Calculate a 0–100 data quality score with transparent penalty breakdown.

    Returns
    -------
    score : int
    breakdown : dict  {penalty_name: penalty_amount}
    """
    base = 100.0
    breakdown: Dict[str, float] = {}
    total_rows = max(len(df), 1)
    total_cells = max(df.size, 1)

    # ── 1. Missing values penalty ─────────────────────────────────────────────
    total_missing = int(df.isna().sum().sum())
    missing_pct = total_missing / total_cells
    missing_penalty = min(missing_pct * 40, 30)  # max 30 pts
    breakdown["missing_values"] = -round(missing_penalty, 2)

    # ── 2. Duplicate penalty ──────────────────────────────────────────────────
    dup_info = detect_duplicates(df)
    dup_pct = dup_info["percentage"] / 100
    dup_penalty = min(dup_pct * 20, 15)
    breakdown["duplicates"] = -round(dup_penalty, 2)

    # ── 3. Outlier penalty ────────────────────────────────────────────────────
    outlier_results = detect_outliers(df)
    if outlier_results:
        avg_outlier_pct = np.mean([r["percentage"] for r in outlier_results]) / 100
        outlier_penalty = min(avg_outlier_pct * 20, 10)
    else:
        outlier_penalty = 0.0
    breakdown["outliers"] = -round(outlier_penalty, 2)

    # ── 4. Data type penalty ──────────────────────────────────────────────────
    object_cols = df.select_dtypes(include=["object"]).columns
    dtype_issues = 0
    for col in object_cols:
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue
        coerced = pd.to_numeric(series.str.replace(",", "").str.replace("$", ""), errors="coerce")
        if coerced.notna().mean() > 0.85:
            dtype_issues += 1
    dtype_penalty = min(dtype_issues * 2, 10)
    breakdown["data_type_issues"] = -round(dtype_penalty, 2)

    # ── 5. Formatting penalty ─────────────────────────────────────────────────
    whitespace_cols = 0
    for col in df.select_dtypes(include=["object"]).columns:
        s = df[col].dropna().astype(str)
        if (s != s.str.strip()).any():
            whitespace_cols += 1
    fmt_penalty = min(whitespace_cols * 1, 5)
    breakdown["formatting"] = -round(fmt_penalty, 2)

    # ── 6. Invalid value penalty ──────────────────────────────────────────────
    # (simple heuristic — negative values in obviously non-negative columns)
    invalid_penalty = 0.0
    for col in df.select_dtypes(include=[np.number]).columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ("age", "count", "qty", "quantity", "amount")):
            n_neg = int((df[col] < 0).sum())
            if n_neg > 0:
                invalid_penalty += min(n_neg / total_rows * 10, 5)
    breakdown["invalid_values"] = -round(min(invalid_penalty, 10), 2)

    # ── Compute final score ───────────────────────────────────────────────────
    total_penalty = sum(abs(v) for v in breakdown.values())
    score = max(0, int(round(base - total_penalty)))

    return score, breakdown
