"""
CleanSense AI — Detector Service

Modular issue-detection functions, each operating on a Pandas DataFrame.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================================
# Helpers
# ============================================================================

def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _classify_column(series: pd.Series) -> str:
    """Return one of: numerical | categorical | boolean | datetime | text"""
    # Boolean
    if series.dtype == bool or set(series.dropna().unique()).issubset({True, False, 0, 1}):
        return "boolean"
    # Datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    # Numerical
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    # Try to coerce to datetime
    try:
        converted = pd.to_datetime(series.dropna().astype(str), infer_datetime_format=True, errors="coerce")
        if converted.notna().mean() > 0.8:
            return "datetime"
    except Exception:
        pass
    # Categorical vs text (by cardinality)
    unique_ratio = series.nunique() / max(len(series), 1)
    if unique_ratio < 0.5 and series.nunique() < 100:
        return "categorical"
    return "text"


# ============================================================================
# 1. Missing Values
# ============================================================================

def detect_missing_values(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts, one per column that has missing values.
    """
    results = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        pct = round(n_missing / len(df) * 100, 2)
        col_type = _classify_column(df[col])
        results.append(
            {
                "column": col,
                "n_missing": n_missing,
                "percentage": pct,
                "col_type": col_type,
            }
        )
    return results


# ============================================================================
# 2. Duplicates
# ============================================================================

def detect_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect fully duplicate rows and potential key columns."""
    dup_mask = df.duplicated(keep=False)
    dup_count = int(df.duplicated().sum())
    pct = round(dup_count / max(len(df), 1) * 100, 2)

    # Potential unique-key columns (high cardinality, object/int type)
    candidate_keys: List[str] = []
    for col in df.columns:
        if df[col].nunique() == len(df) and df[col].notna().all():
            candidate_keys.append(col)

    # Near-duplicate key columns (appear multiple times)
    near_dup_keys: List[Dict] = []
    for col in df.columns:
        dup_in_col = int(df[col].duplicated().sum())
        if 0 < dup_in_col < len(df) * 0.05:
            near_dup_keys.append({"column": col, "duplicated_values": dup_in_col})

    return {
        "duplicate_rows": dup_count,
        "percentage": pct,
        "candidate_keys": candidate_keys,
        "near_dup_keys": near_dup_keys,
    }


# ============================================================================
# 3. Outliers
# ============================================================================

def detect_outliers(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detect outliers using IQR, Z-score, and Modified Z-score methods.
    Returns one dict per numerical column with outliers.
    """
    results = []
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    for col in numerical_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue

        # IQR method
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_outliers = series[(series < lower) | (series > upper)]

        # Modified Z-score (median-based, robust)
        median = series.median()
        mad = np.median(np.abs(series - median))
        if mad > 0:
            mod_z = 0.6745 * (series - median) / mad
            mz_outliers = series[np.abs(mod_z) > 3.5]
        else:
            mz_outliers = pd.Series([], dtype=series.dtype)

        n_outliers = int(max(len(iqr_outliers), len(mz_outliers)))
        if n_outliers == 0:
            continue

        outlier_values = sorted(series[(series < lower) | (series > upper)].unique().tolist())
        pct = round(n_outliers / len(series) * 100, 2)

        results.append(
            {
                "column": col,
                "n_outliers": n_outliers,
                "percentage": pct,
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
                "min_val": round(float(series.min()), 4),
                "max_val": round(float(series.max()), 4),
                "outlier_values": outlier_values[:10],  # sample
                "method": "IQR + Modified Z-Score",
            }
        )
    return results


# ============================================================================
# 4. Data Type Issues
# ============================================================================

def detect_data_type_issues(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detect columns stored as wrong dtype (numeric-as-string, dates-as-string, etc.)"""
    results = []
    object_cols = df.select_dtypes(include=["object"]).columns

    for col in object_cols:
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue

        # Numeric stored as string
        numeric_pattern = re.compile(r"^-?\s*[\$€£¥]?\s*[\d,]+\.?\d*\s*%?$")
        currency_pattern = re.compile(r"^[\$€£¥]\s*[\d,]+\.?\d*$")
        pct_pattern = re.compile(r"^-?\d+\.?\d*\s*%$")

        numeric_matches = series.apply(lambda x: bool(numeric_pattern.match(x.strip()))).mean()
        currency_matches = series.apply(lambda x: bool(currency_pattern.match(x.strip()))).mean()
        pct_matches = series.apply(lambda x: bool(pct_pattern.match(x.strip()))).mean()

        if currency_matches > 0.7:
            sample = series.head(5).tolist()
            results.append(
                {
                    "column": col,
                    "current_dtype": str(df[col].dtype),
                    "detected_format": "currency",
                    "recommendation": "remove_currency_symbol_and_convert_numeric",
                    "sample_values": sample,
                    "match_percentage": round(currency_matches * 100, 1),
                }
            )
            continue

        if pct_matches > 0.7:
            sample = series.head(5).tolist()
            results.append(
                {
                    "column": col,
                    "current_dtype": str(df[col].dtype),
                    "detected_format": "percentage",
                    "recommendation": "remove_percent_symbol_and_convert_numeric",
                    "sample_values": sample,
                    "match_percentage": round(pct_matches * 100, 1),
                }
            )
            continue

        if numeric_matches > 0.85:
            coerced = pd.to_numeric(series.str.replace(",", ""), errors="coerce")
            if coerced.notna().mean() > 0.85:
                is_int = (coerced.dropna() % 1 == 0).all()
                sample = series.head(5).tolist()
                results.append(
                    {
                        "column": col,
                        "current_dtype": str(df[col].dtype),
                        "detected_format": "integer" if is_int else "float",
                        "recommendation": "convert_to_integer" if is_int else "convert_to_float",
                        "sample_values": sample,
                        "match_percentage": round(numeric_matches * 100, 1),
                    }
                )
                continue

        # Date stored as string
        try:
            converted = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
            date_ratio = converted.notna().mean()
            if date_ratio > 0.8:
                sample = series.head(5).tolist()
                results.append(
                    {
                        "column": col,
                        "current_dtype": str(df[col].dtype),
                        "detected_format": "datetime",
                        "recommendation": "convert_to_datetime",
                        "sample_values": sample,
                        "match_percentage": round(date_ratio * 100, 1),
                    }
                )
                continue
        except Exception:
            pass

        # Boolean stored as string
        bool_values = {"true", "false", "yes", "no", "1", "0", "y", "n", "t", "f"}
        unique_lower = set(series.str.lower().unique())
        if unique_lower.issubset(bool_values) and len(unique_lower) <= 4:
            results.append(
                {
                    "column": col,
                    "current_dtype": str(df[col].dtype),
                    "detected_format": "boolean",
                    "recommendation": "convert_to_boolean",
                    "sample_values": series.head(5).tolist(),
                    "match_percentage": 100.0,
                }
            )

    return results


# ============================================================================
# 5. String Inconsistencies
# ============================================================================

def detect_string_inconsistencies(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detect leading/trailing spaces, mixed case, extra spaces in text/categorical columns."""
    results = []
    str_cols = df.select_dtypes(include=["object"]).columns

    for col in str_cols:
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue

        issues = []
        # Leading / trailing whitespace
        n_whitespace = int((series != series.str.strip()).sum())
        if n_whitespace > 0:
            issues.append({"type": "leading_trailing_spaces", "count": n_whitespace})

        # Mixed case (if cardinality suggests categorical)
        unique_raw = series.unique()
        unique_lower = series.str.lower().unique()
        if len(unique_raw) > len(unique_lower):
            n_case_issues = len(unique_raw) - len(unique_lower)
            issues.append({"type": "mixed_case", "count": n_case_issues})
            # Show examples of same value in different cases
            case_examples = []
            seen_lower = {}
            for v in unique_raw:
                lv = v.lower().strip()
                if lv in seen_lower:
                    if len(case_examples) < 5:
                        case_examples.append([seen_lower[lv], v])
                else:
                    seen_lower[lv] = v
            if case_examples:
                issues[-1]["examples"] = case_examples

        # Extra internal spaces
        n_extra = int((series.str.replace(r"\s+", " ", regex=True) != series).sum())
        if n_extra > 0:
            issues.append({"type": "extra_spaces", "count": n_extra})

        if issues:
            results.append(
                {
                    "column": col,
                    "issues": issues,
                    "sample_values": series.unique()[:8].tolist(),
                }
            )

    return results


# ============================================================================
# 6. Constant / Near-Constant Columns
# ============================================================================

def detect_constant_columns(df: pd.DataFrame, near_constant_threshold: float = 0.98) -> List[Dict[str, Any]]:
    """Detect constant (1 unique value) and near-constant columns."""
    results = []
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        dominant = df[col].value_counts(normalize=True, dropna=True)
        top_pct = float(dominant.iloc[0]) if len(dominant) > 0 else 0.0

        if n_unique == 1:
            results.append(
                {
                    "column": col,
                    "type": "constant",
                    "unique_values": 1,
                    "dominant_value": str(dominant.index[0]) if len(dominant) else "N/A",
                    "dominant_percentage": 100.0,
                }
            )
        elif top_pct >= near_constant_threshold:
            results.append(
                {
                    "column": col,
                    "type": "near_constant",
                    "unique_values": n_unique,
                    "dominant_value": str(dominant.index[0]),
                    "dominant_percentage": round(top_pct * 100, 2),
                }
            )
    return results


# ============================================================================
# 7. High-Missing Columns
# ============================================================================

def detect_high_missing_columns(
    df: pd.DataFrame,
    warning_threshold: float = 30.0,
    high_threshold: float = 50.0,
    critical_threshold: float = 80.0,
) -> List[Dict[str, Any]]:
    """Return columns with high missing-value percentages."""
    results = []
    for col in df.columns:
        pct = df[col].isna().mean() * 100
        if pct < warning_threshold:
            continue
        level = "warning"
        if pct >= critical_threshold:
            level = "critical"
        elif pct >= high_threshold:
            level = "high"
        results.append(
            {
                "column": col,
                "missing_percentage": round(pct, 2),
                "missing_count": int(df[col].isna().sum()),
                "level": level,
            }
        )
    return results


# ============================================================================
# 8. Invalid Values
# ============================================================================

def detect_invalid_values(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Apply domain-logic heuristics to detect suspicious/invalid values.
    Examples: negative age, emails, out-of-range percentages.
    """
    results = []

    AGE_COLS = re.compile(r"\bage\b", re.I)
    EMAIL_COLS = re.compile(r"\bemail\b", re.I)
    PCT_COLS = re.compile(r"\bpercent(age)?\b|\bpct\b|\brate\b", re.I)
    ID_COLS = re.compile(r"\bid\b", re.I)
    PHONE_COLS = re.compile(r"\bphone\b|\bmobile\b|\bcontact\b", re.I)

    for col in df.columns:
        series = df[col].dropna()
        col_issues: List[Dict] = []

        # ── Age checks ──────────────────────────────────────────────────────
        if AGE_COLS.search(col) and pd.api.types.is_numeric_dtype(series):
            invalid = series[(series < 0) | (series > 120)]
            if len(invalid):
                col_issues.append(
                    {
                        "type": "invalid_age",
                        "count": int(len(invalid)),
                        "sample_values": invalid.unique()[:5].tolist(),
                        "rule": "Age must be 0–120",
                    }
                )

        # ── Email checks ─────────────────────────────────────────────────────
        if EMAIL_COLS.search(col):
            email_re = re.compile(r"^[\w._%+\-]+@[\w.\-]+\.[A-Za-z]{2,}$")
            invalid_emails = series.astype(str).apply(lambda x: not bool(email_re.match(x.strip())))
            n_invalid = int(invalid_emails.sum())
            if n_invalid:
                col_issues.append(
                    {
                        "type": "invalid_email",
                        "count": n_invalid,
                        "sample_values": series.astype(str)[invalid_emails].head(5).tolist(),
                        "rule": "Must match standard email format",
                    }
                )

        # ── Percentage checks ────────────────────────────────────────────────
        if PCT_COLS.search(col) and pd.api.types.is_numeric_dtype(series):
            if series.max() <= 1.01:  # likely 0–1 scale
                invalid = series[(series < 0) | (series > 1)]
            else:
                invalid = series[(series < 0) | (series > 100)]
            if len(invalid):
                col_issues.append(
                    {
                        "type": "invalid_percentage",
                        "count": int(len(invalid)),
                        "sample_values": invalid.unique()[:5].tolist(),
                        "rule": "Percentage must be 0–100 (or 0–1)",
                    }
                )

        if col_issues:
            results.append({"column": col, "issues": col_issues})

    return results


# ============================================================================
# 9. Date Issues
# ============================================================================

def detect_date_issues(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detect date columns with format inconsistencies, invalid dates, future dates."""
    results = []
    import datetime

    today = pd.Timestamp.now()

    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        col_type = _classify_column(df[col])
        if col_type not in ("datetime", "text", "categorical"):
            continue

        # Attempt parsing
        parsed = pd.to_datetime(series.astype(str), infer_datetime_format=True, errors="coerce")
        parse_ratio = parsed.notna().mean()

        if parse_ratio < 0.5:
            continue  # Not a date column

        issues = []

        # Invalid (failed to parse)
        n_invalid = int(parsed.isna().sum())
        if n_invalid > 0:
            issues.append({"type": "unparseable_dates", "count": n_invalid})

        # Multiple formats
        if pd.api.types.is_object_dtype(series):
            sample_str = series.head(100).astype(str)
            fmt_guesses = set()
            common_fmts = ["%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%B %d, %Y"]
            for fmt in common_fmts:
                ok = sample_str.apply(lambda x: _try_fmt(x, fmt)).sum()
                if ok > 5:
                    fmt_guesses.add(fmt)
            if len(fmt_guesses) > 1:
                issues.append({"type": "multiple_formats", "formats_detected": list(fmt_guesses)})

        # Future dates
        future = parsed[parsed > today]
        if len(future) > 0:
            issues.append({"type": "future_dates", "count": int(len(future))})

        if issues:
            results.append(
                {
                    "column": col,
                    "parse_success_ratio": round(parse_ratio * 100, 1),
                    "issues": issues,
                    "sample_values": series.head(5).tolist(),
                }
            )

    return results


def _try_fmt(s: str, fmt: str) -> bool:
    try:
        pd.to_datetime(s, format=fmt)
        return True
    except Exception:
        return False


# ============================================================================
# 10. High Cardinality
# ============================================================================

def detect_high_cardinality(df: pd.DataFrame, threshold: int = 50) -> List[Dict[str, Any]]:
    """Detect categorical columns with suspiciously many unique values."""
    results = []
    for col in df.select_dtypes(include=["object"]).columns:
        n_unique = df[col].nunique()
        if n_unique > threshold:
            results.append(
                {
                    "column": col,
                    "unique_values": n_unique,
                    "cardinality_ratio": round(n_unique / max(len(df), 1) * 100, 2),
                }
            )
    return results


# ============================================================================
# 11. Categorical Inconsistencies
# ============================================================================

def detect_categorical_inconsistencies(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detect categorical columns where the same category appears
    with different spellings / capitalisations.
    """
    results = []
    for col in df.select_dtypes(include=["object"]).columns:
        series = df[col].dropna().astype(str)
        unique_vals = series.unique()

        if len(unique_vals) > 200 or len(unique_vals) < 2:
            continue

        # Group by lowercased+stripped version
        groups: Dict[str, List[str]] = {}
        for v in unique_vals:
            key = v.lower().strip()
            groups.setdefault(key, []).append(v)

        ambiguous = {k: v for k, v in groups.items() if len(v) > 1}

        if ambiguous:
            results.append(
                {
                    "column": col,
                    "inconsistency_groups": [
                        {"canonical": k, "variants": v}
                        for k, v in list(ambiguous.items())[:10]
                    ],
                    "n_affected_groups": len(ambiguous),
                }
            )

    return results


# ============================================================================
# 12. Correlation
# ============================================================================

def detect_correlation(df: pd.DataFrame, threshold: float = 0.9) -> List[Dict[str, Any]]:
    """Detect highly correlated numerical column pairs."""
    numerical = df.select_dtypes(include=[np.number])
    if numerical.shape[1] < 2:
        return []

    corr = numerical.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    pairs = []
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            if pd.notna(val) and val >= threshold:
                pairs.append(
                    {
                        "col1": idx,
                        "col2": col,
                        "correlation": round(float(val), 4),
                    }
                )

    return sorted(pairs, key=lambda x: x["correlation"], reverse=True)


# ============================================================================
# Master column profile builder
# ============================================================================

def build_column_profile(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build a comprehensive profile for every column."""
    profiles = []
    for col in df.columns:
        series = df[col]
        col_type = _classify_column(series)
        non_null = series.dropna()
        n_null = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        pct_null = round(n_null / max(len(series), 1) * 100, 2)

        # Cardinality label
        ratio = n_unique / max(len(non_null), 1)
        cardinality = "low" if ratio < 0.05 else ("high" if ratio > 0.5 else "medium")

        profile: Dict[str, Any] = {
            "name": col,
            "dtype": str(series.dtype),
            "category": col_type,
            "non_null_count": int(len(non_null)),
            "null_count": n_null,
            "null_percentage": pct_null,
            "unique_count": n_unique,
            "cardinality": cardinality,
            "is_constant": n_unique == 1,
            "is_near_constant": False,
            "sample_values": non_null.head(5).tolist(),
        }

        # Near-constant
        if n_unique > 1:
            top_pct = float(series.value_counts(normalize=True, dropna=True).iloc[0])
            profile["is_near_constant"] = top_pct >= 0.98

        # Numerical extras
        if col_type == "numerical":
            try:
                profile["mean"] = round(float(non_null.mean()), 4)
                profile["median"] = round(float(non_null.median()), 4)
                mode_vals = non_null.mode()
                profile["mode"] = float(mode_vals.iloc[0]) if len(mode_vals) else None
                profile["std"] = round(float(non_null.std()), 4)
                profile["variance"] = round(float(non_null.var()), 4)
                profile["min_val"] = round(float(non_null.min()), 4)
                profile["max_val"] = round(float(non_null.max()), 4)
                profile["q25"] = round(float(non_null.quantile(0.25)), 4)
                profile["q75"] = round(float(non_null.quantile(0.75)), 4)
                profile["iqr"] = round(float(non_null.quantile(0.75) - non_null.quantile(0.25)), 4)
                profile["skewness"] = round(float(non_null.skew()), 4)

                # IQR outliers
                q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    out = non_null[(non_null < q1 - 1.5 * iqr) | (non_null > q3 + 1.5 * iqr)]
                    profile["outlier_count"] = int(len(out))
                    profile["outlier_percentage"] = round(len(out) / max(len(non_null), 1) * 100, 2)
            except Exception:
                pass

        profiles.append(profile)
    return profiles
