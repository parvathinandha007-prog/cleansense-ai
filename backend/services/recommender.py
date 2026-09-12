"""
CleanSense AI — Recommendation Engine

Rule-based engine that transforms raw detection results into
structured, human-readable recommendations with severity levels.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List

import pandas as pd

from services.detector import (
    detect_missing_values,
    detect_duplicates,
    detect_outliers,
    detect_data_type_issues,
    detect_string_inconsistencies,
    detect_constant_columns,
    detect_high_missing_columns,
    detect_invalid_values,
    detect_date_issues,
    detect_high_cardinality,
    detect_categorical_inconsistencies,
    detect_correlation,
    _classify_column,
)


# ── Severity helpers ─────────────────────────────────────────────────────────

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _sev(pct: float, count: int, total: int) -> str:
    if pct >= 50 or count >= total * 0.5:
        return "critical"
    if pct >= 20 or count >= total * 0.2:
        return "high"
    if pct >= 5 or count >= total * 0.05:
        return "medium"
    return "low"


# ── Missing value recommendation logic ───────────────────────────────────────

def _missing_action(col: str, pct: float, col_type: str, df: pd.DataFrame) -> Dict[str, str]:
    """Choose the best imputation strategy for a missing-value problem."""

    if pct >= 80:
        return {
            "recommendation": "drop_column",
            "reason": (
                f"{pct:.1f}% of values are missing. "
                "This column contains very little useful information and "
                "is likely not worth imputing."
            ),
        }

    if pct >= 40:
        return {
            "recommendation": "drop_column_or_leave",
            "reason": (
                f"{pct:.1f}% of values are missing. "
                "The high missing rate makes imputation unreliable. "
                "Consider removing or using domain knowledge."
            ),
        }

    if col_type == "numerical":
        try:
            skew = float(df[col].dropna().skew())
        except Exception:
            skew = 0.0

        if abs(skew) > 1:
            return {
                "recommendation": "median_imputation",
                "reason": (
                    f"Numerical column with skewness {skew:.2f}. "
                    "Median is more robust to skewed distributions and outliers than mean."
                ),
            }
        return {
            "recommendation": "mean_imputation",
            "reason": (
                "Numerical column with approximately symmetric distribution. "
                "Mean imputation is appropriate."
            ),
        }

    if col_type == "categorical":
        return {
            "recommendation": "mode_imputation",
            "reason": "Categorical column — filling with the most frequent category (mode).",
        }

    if col_type == "datetime":
        return {
            "recommendation": "forward_fill",
            "reason": "Date/time column — forward-fill propagates the last known date.",
        }

    return {
        "recommendation": "leave_unchanged",
        "reason": "Unable to determine a safe automatic strategy. Manual review recommended.",
    }


# ── Main recommendation generator ────────────────────────────────────────────

def generate_recommendations(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Run all detectors and produce a prioritised list of recommendations.
    """
    total_rows = len(df)
    recs: List[Dict[str, Any]] = []

    # ── 1. High-missing columns (check before general missing) ───────────────
    high_missing = detect_high_missing_columns(df)
    for item in high_missing:
        col = item["column"]
        pct = item["missing_percentage"]
        level_map = {"critical": "critical", "high": "high", "warning": "medium"}
        sev = level_map.get(item["level"], "medium")
        recs.append(
            {
                "id": str(uuid.uuid4()),
                "column": col,
                "issue_type": "high_missing_column",
                "severity": sev,
                "title": f"High missing rate in '{col}'",
                "description": f"{pct:.1f}% of values are missing ({item['missing_count']:,} rows).",
                "count": item["missing_count"],
                "percentage": pct,
                "recommendation": "drop_column",
                "reason": (
                    f"With {pct:.1f}% missing values, imputation would introduce "
                    "significant bias. Dropping this column is recommended unless "
                    "domain knowledge suggests otherwise."
                ),
                "possible_actions": [
                    "drop_column",
                    "median_imputation",
                    "mean_imputation",
                    "mode_imputation",
                    "constant_value",
                    "leave_unchanged",
                ],
                "default_action": "drop_column",
                "requires_user_approval": True,
                "extra_data": {"missing_level": item["level"]},
            }
        )

    high_missing_cols = {i["column"] for i in high_missing}

    # ── 2. Missing values (regular columns) ──────────────────────────────────
    missing = detect_missing_values(df)
    for item in missing:
        col = item["column"]
        if col in high_missing_cols:
            continue  # Already handled above
        pct = item["percentage"]
        col_type = item["col_type"]
        sev = _sev(pct, item["n_missing"], total_rows)
        action_info = _missing_action(col, pct, col_type, df)
        recs.append(
            {
                "id": str(uuid.uuid4()),
                "column": col,
                "issue_type": "missing_values",
                "severity": sev,
                "title": f"Missing values in '{col}'",
                "description": f"{item['n_missing']:,} missing values ({pct:.1f}% of rows).",
                "count": item["n_missing"],
                "percentage": pct,
                "recommendation": action_info["recommendation"],
                "reason": action_info["reason"],
                "possible_actions": [
                    "drop_rows",
                    "drop_column",
                    "mean_imputation",
                    "median_imputation",
                    "mode_imputation",
                    "forward_fill",
                    "backward_fill",
                    "constant_value",
                    "leave_unchanged",
                ],
                "default_action": action_info["recommendation"],
                "requires_user_approval": True,
                "extra_data": {"col_type": col_type},
            }
        )

    # ── 3. Duplicates ─────────────────────────────────────────────────────────
    dup_info = detect_duplicates(df)
    if dup_info["duplicate_rows"] > 0:
        n = dup_info["duplicate_rows"]
        pct = dup_info["percentage"]
        sev = _sev(pct, n, total_rows)
        recs.append(
            {
                "id": str(uuid.uuid4()),
                "column": None,
                "issue_type": "duplicate_rows",
                "severity": sev,
                "title": "Duplicate rows detected",
                "description": f"{n:,} exact duplicate rows found ({pct:.1f}% of dataset).",
                "count": n,
                "percentage": pct,
                "recommendation": "drop_duplicates_keep_first",
                "reason": (
                    "Duplicate rows introduce data bias and inflate statistics. "
                    "Keeping the first occurrence is the safest default."
                ),
                "possible_actions": [
                    "drop_duplicates_keep_first",
                    "drop_duplicates_keep_last",
                    "drop_duplicates_all",
                    "leave_unchanged",
                ],
                "default_action": "drop_duplicates_keep_first",
                "requires_user_approval": True,
                "extra_data": {
                    "candidate_keys": dup_info["candidate_keys"],
                    "near_dup_keys": dup_info["near_dup_keys"],
                },
            }
        )

    # ── 4. Outliers ───────────────────────────────────────────────────────────
    outliers = detect_outliers(df)
    for item in outliers:
        col = item["column"]
        pct = item["percentage"]
        n = item["n_outliers"]
        sev = "high" if pct > 10 else ("medium" if pct > 2 else "low")

        # Domain check: are values physically impossible?
        reason_suffix = ""
        if any(abs(v) > 1e6 for v in (item.get("outlier_values") or [])):
            reason_suffix = " Some values appear physically implausible."

        recs.append(
            {
                "id": str(uuid.uuid4()),
                "column": col,
                "issue_type": "outliers",
                "severity": sev,
                "title": f"Potential outliers in '{col}'",
                "description": (
                    f"{n:,} potential outliers detected ({pct:.1f}%). "
                    f"Bounds: [{item['lower_bound']}, {item['upper_bound']}]."
                ),
                "count": n,
                "percentage": pct,
                "recommendation": "review_outliers",
                "reason": (
                    "Outliers are not necessarily errors — they may represent "
                    "legitimate extreme values. Review before removing." + reason_suffix
                ),
                "possible_actions": [
                    "remove_outliers",
                    "cap_winsorize",
                    "replace_with_median",
                    "review_manually",
                    "leave_unchanged",
                ],
                "default_action": "review_manually",
                "requires_user_approval": True,
                "extra_data": {
                    "outlier_values": item.get("outlier_values", []),
                    "lower_bound": item["lower_bound"],
                    "upper_bound": item["upper_bound"],
                    "method": item["method"],
                },
            }
        )

    # ── 5. Data type issues ───────────────────────────────────────────────────
    dtype_issues = detect_data_type_issues(df)
    for item in dtype_issues:
        col = item["column"]
        fmt = item["detected_format"]
        recs.append(
            {
                "id": str(uuid.uuid4()),
                "column": col,
                "issue_type": "data_type",
                "severity": "medium",
                "title": f"Type mismatch in '{col}'",
                "description": (
                    f"Column stored as '{item['current_dtype']}' "
                    f"but appears to contain {fmt} values."
                ),
                "count": None,
                "percentage": item.get("match_percentage"),
                "recommendation": item["recommendation"],
                "reason": (
                    f"Converting to the correct type enables proper "
                    f"mathematical operations and reduces memory usage."
                ),
                "possible_actions": [
                    item["recommendation"],
                    "leave_unchanged",
                ],
                "default_action": item["recommendation"],
                "requires_user_approval": True,
                "extra_data": {"sample_values": item.get("sample_values", [])},
            }
        )

    # ── 6. String inconsistencies ─────────────────────────────────────────────
    str_issues = detect_string_inconsistencies(df)
    for item in str_issues:
        col = item["column"]
        issue_summary = ", ".join(i["type"].replace("_", " ") for i in item["issues"])
        recs.append(
            {
                "id": str(uuid.uuid4()),
                "column": col,
                "issue_type": "string_inconsistency",
                "severity": "low",
                "title": f"String formatting issues in '{col}'",
                "description": f"Issues found: {issue_summary}.",
                "count": sum(i.get("count", 0) for i in item["issues"]),
                "percentage": None,
                "recommendation": "strip_and_normalize",
                "reason": (
                    "Inconsistent formatting causes incorrect grouping, "
                    "joins, and downstream analysis errors."
                ),
                "possible_actions": [
                    "strip_and_normalize",
                    "to_lower",
                    "to_upper",
                    "to_title_case",
                    "leave_unchanged",
                ],
                "default_action": "strip_and_normalize",
                "requires_user_approval": True,
                "extra_data": {"issues": item["issues"], "sample_values": item["sample_values"]},
            }
        )

    # ── 7. Constant / near-constant columns ───────────────────────────────────
    const_cols = detect_constant_columns(df)
    for item in const_cols:
        col = item["column"]
        is_const = item["type"] == "constant"
        recs.append(
            {
                "id": str(uuid.uuid4()),
                "column": col,
                "issue_type": "constant_column",
                "severity": "low",
                "title": f"{'Constant' if is_const else 'Near-constant'} column '{col}'",
                "description": (
                    f"Dominant value: '{item['dominant_value']}' "
                    f"({item['dominant_percentage']:.1f}% of rows)."
                ),
                "count": item["unique_values"],
                "percentage": item["dominant_percentage"],
                "recommendation": "drop_column",
                "reason": (
                    "Constant or near-constant columns provide no useful "
                    "variation for analysis or modeling."
                ),
                "possible_actions": ["drop_column", "leave_unchanged"],
                "default_action": "drop_column",
                "requires_user_approval": True,
                "extra_data": {"column_type": item["type"]},
            }
        )

    # ── 8. Invalid values ─────────────────────────────────────────────────────
    invalid = detect_invalid_values(df)
    for item in invalid:
        col = item["column"]
        for issue in item["issues"]:
            recs.append(
                {
                    "id": str(uuid.uuid4()),
                    "column": col,
                    "issue_type": "invalid_values",
                    "severity": "high",
                    "title": f"Invalid values in '{col}'",
                    "description": (
                        f"{issue['count']} values violate rule: {issue['rule']}. "
                        f"Sample: {issue.get('sample_values', [])}"
                    ),
                    "count": issue["count"],
                    "percentage": round(issue["count"] / max(total_rows, 1) * 100, 2),
                    "recommendation": "review_and_remove",
                    "reason": (
                        "Values violate expected domain constraints. "
                        "They may be data entry errors or corrupt records."
                    ),
                    "possible_actions": [
                        "remove_invalid_rows",
                        "replace_with_median",
                        "replace_with_nan",
                        "review_manually",
                        "leave_unchanged",
                    ],
                    "default_action": "review_manually",
                    "requires_user_approval": True,
                    "extra_data": {
                        "rule": issue["rule"],
                        "sample_values": issue.get("sample_values", []),
                        "issue_type": issue["type"],
                    },
                }
            )

    # ── 9. Date issues ────────────────────────────────────────────────────────
    date_issues = detect_date_issues(df)
    for item in date_issues:
        col = item["column"]
        recs.append(
            {
                "id": str(uuid.uuid4()),
                "column": col,
                "issue_type": "date_format",
                "severity": "medium",
                "title": f"Date format issues in '{col}'",
                "description": (
                    f"Parse success: {item['parse_success_ratio']:.1f}%. "
                    f"Issues: {', '.join(i['type'].replace('_', ' ') for i in item['issues'])}."
                ),
                "count": None,
                "percentage": None,
                "recommendation": "standardize_datetime",
                "reason": (
                    "Inconsistent date formats cause incorrect sorting, "
                    "time-series analysis, and join failures."
                ),
                "possible_actions": ["standardize_datetime", "leave_unchanged"],
                "default_action": "standardize_datetime",
                "requires_user_approval": True,
                "extra_data": {"issues": item["issues"], "sample_values": item["sample_values"]},
            }
        )

    # ── 10. Categorical inconsistencies ──────────────────────────────────────
    cat_issues = detect_categorical_inconsistencies(df)
    for item in cat_issues:
        col = item["column"]
        recs.append(
            {
                "id": str(uuid.uuid4()),
                "column": col,
                "issue_type": "categorical_inconsistency",
                "severity": "medium",
                "title": f"Inconsistent categories in '{col}'",
                "description": (
                    f"{item['n_affected_groups']} category group(s) with "
                    "multiple spelling/capitalisation variants."
                ),
                "count": item["n_affected_groups"],
                "percentage": None,
                "recommendation": "standardize_categories",
                "reason": (
                    "Multiple representations of the same category cause "
                    "incorrect group counts and skewed analytics."
                ),
                "possible_actions": [
                    "standardize_categories",
                    "to_lower",
                    "to_upper",
                    "to_title_case",
                    "leave_unchanged",
                ],
                "default_action": "standardize_categories",
                "requires_user_approval": True,
                "extra_data": {"inconsistency_groups": item["inconsistency_groups"]},
            }
        )

    # ── 11. Correlation ───────────────────────────────────────────────────────
    corr_pairs = detect_correlation(df)
    for pair in corr_pairs[:5]:  # limit to top 5
        recs.append(
            {
                "id": str(uuid.uuid4()),
                "column": f"{pair['col1']} ↔ {pair['col2']}",
                "issue_type": "high_correlation",
                "severity": "low",
                "title": f"High correlation: '{pair['col1']}' ↔ '{pair['col2']}'",
                "description": f"Pearson correlation = {pair['correlation']:.4f}.",
                "count": None,
                "percentage": None,
                "recommendation": "review_for_multicollinearity",
                "reason": (
                    "High correlation may indicate redundant features. "
                    "For ML models, consider removing one. "
                    "Correlation alone does not imply causation or redundancy."
                ),
                "possible_actions": [
                    "drop_column",
                    "review_manually",
                    "leave_unchanged",
                ],
                "default_action": "leave_unchanged",
                "requires_user_approval": True,
                "extra_data": pair,
            }
        )

    # ── Sort by severity ──────────────────────────────────────────────────────
    recs.sort(key=lambda r: SEVERITY_ORDER.get(r["severity"], 99))

    return recs
