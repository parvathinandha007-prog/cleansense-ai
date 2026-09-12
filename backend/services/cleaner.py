"""
CleanSense AI — Cleaning Engine

Applies user-approved operations in a safe, ordered sequence.
Maintains an audit log and supports undo via DataFrame snapshots.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ── Operation executor ────────────────────────────────────────────────────────

def apply_cleaning_operations(
    df: pd.DataFrame,
    operations: List[Dict[str, Any]],
) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Apply a list of cleaning operations to the DataFrame.

    Operations are applied in a logical order regardless of input order:
      1. Standardise missing values
      2. Remove duplicates
      3. Fix data types
      4. Clean strings
      5. Handle missing values
      6. Handle invalid values
      7. Handle outliers

    Returns
    -------
    cleaned_df : pd.DataFrame
    audit_log  : list of result dicts
    """
    ORDER = {
        "standardize_missing": 0,
        "drop_duplicates_keep_first": 1,
        "drop_duplicates_keep_last": 1,
        "drop_duplicates_all": 1,
        "convert_to_integer": 2,
        "convert_to_float": 2,
        "convert_to_datetime": 2,
        "convert_to_boolean": 2,
        "remove_currency_symbol_and_convert_numeric": 2,
        "remove_percent_symbol_and_convert_numeric": 2,
        "standardize_datetime": 2,
        "strip_and_normalize": 3,
        "to_lower": 3,
        "to_upper": 3,
        "to_title_case": 3,
        "standardize_categories": 3,
        "drop_column": 4,
        "drop_rows": 5,
        "mean_imputation": 5,
        "median_imputation": 5,
        "mode_imputation": 5,
        "forward_fill": 5,
        "backward_fill": 5,
        "constant_value": 5,
        "remove_invalid_rows": 6,
        "replace_with_nan": 6,
        "replace_with_median": 6,
        "remove_outliers": 7,
        "cap_winsorize": 7,
        "leave_unchanged": 99,
        "review_manually": 99,
        "review_for_multicollinearity": 99,
        "drop_column_or_leave": 99,
        "review_and_remove": 99,
    }

    ops_sorted = sorted(operations, key=lambda o: ORDER.get(o.get("action", ""), 50))
    audit_log: List[Dict[str, Any]] = []
    result = df.copy()

    for op in ops_sorted:
        action = op.get("action", "")
        col = op.get("column")
        rec_id = op.get("recommendation_id", "")

        try:
            rows_before = len(result)

            if action == "leave_unchanged" or action in (
                "review_manually", "review_for_multicollinearity",
                "review_and_remove", "drop_column_or_leave",
            ):
                audit_log.append(_log(rec_id, action, col, "skipped", "No change requested by user."))
                continue

            # ── Duplicates ────────────────────────────────────────────────
            if action == "drop_duplicates_keep_first":
                subset = op.get("subset_columns")
                result = result.drop_duplicates(subset=subset, keep="first")
                removed = rows_before - len(result)
                audit_log.append(_log(rec_id, action, col, "applied", f"Removed {removed} duplicate rows (keep first)."))

            elif action == "drop_duplicates_keep_last":
                subset = op.get("subset_columns")
                result = result.drop_duplicates(subset=subset, keep="last")
                removed = rows_before - len(result)
                audit_log.append(_log(rec_id, action, col, "applied", f"Removed {removed} duplicate rows (keep last)."))

            elif action == "drop_duplicates_all":
                subset = op.get("subset_columns")
                result = result.drop_duplicates(subset=subset, keep=False)
                removed = rows_before - len(result)
                audit_log.append(_log(rec_id, action, col, "applied", f"Removed all {removed} duplicate rows."))

            # ── Drop column ────────────────────────────────────────────────
            elif action == "drop_column":
                if col and col in result.columns:
                    result = result.drop(columns=[col])
                    audit_log.append(_log(rec_id, action, col, "applied", f"Dropped column '{col}'."))
                else:
                    audit_log.append(_log(rec_id, action, col, "skipped", "Column not found."))

            # ── Drop rows with missing ─────────────────────────────────────
            elif action == "drop_rows":
                if col and col in result.columns:
                    mask = result[col].isna()
                    removed = int(mask.sum())
                    result = result[~mask]
                    audit_log.append(_log(rec_id, action, col, "applied", f"Dropped {removed} rows with missing '{col}'."))

            # ── Imputation ─────────────────────────────────────────────────
            elif action == "mean_imputation":
                if col and col in result.columns:
                    fill_val = result[col].mean()
                    n_filled = int(result[col].isna().sum())
                    result[col] = result[col].fillna(fill_val)
                    audit_log.append(_log(rec_id, action, col, "applied", f"Filled {n_filled} missing values with mean ({fill_val:.4f})."))

            elif action == "median_imputation":
                if col and col in result.columns:
                    fill_val = result[col].median()
                    n_filled = int(result[col].isna().sum())
                    result[col] = result[col].fillna(fill_val)
                    audit_log.append(_log(rec_id, action, col, "applied", f"Filled {n_filled} missing values with median ({fill_val:.4f})."))

            elif action == "mode_imputation":
                if col and col in result.columns:
                    mode_vals = result[col].mode()
                    if len(mode_vals):
                        fill_val = mode_vals.iloc[0]
                        n_filled = int(result[col].isna().sum())
                        result[col] = result[col].fillna(fill_val)
                        audit_log.append(_log(rec_id, action, col, "applied", f"Filled {n_filled} missing values with mode ('{fill_val}')."))

            elif action == "forward_fill":
                if col and col in result.columns:
                    n_filled = int(result[col].isna().sum())
                    result[col] = result[col].ffill()
                    audit_log.append(_log(rec_id, action, col, "applied", f"Forward-filled {n_filled} missing values."))

            elif action == "backward_fill":
                if col and col in result.columns:
                    n_filled = int(result[col].isna().sum())
                    result[col] = result[col].bfill()
                    audit_log.append(_log(rec_id, action, col, "applied", f"Backward-filled {n_filled} missing values."))

            elif action == "constant_value":
                if col and col in result.columns:
                    fill_val = op.get("custom_value", 0)
                    n_filled = int(result[col].isna().sum())
                    result[col] = result[col].fillna(fill_val)
                    audit_log.append(_log(rec_id, action, col, "applied", f"Filled {n_filled} missing values with constant '{fill_val}'."))

            # ── Type conversions ──────────────────────────────────────────
            elif action == "convert_to_integer":
                if col and col in result.columns:
                    result[col] = pd.to_numeric(result[col].astype(str).str.replace(",", ""), errors="coerce")
                    result[col] = result[col].astype("Int64")
                    audit_log.append(_log(rec_id, action, col, "applied", f"Converted '{col}' to integer."))

            elif action == "convert_to_float":
                if col and col in result.columns:
                    result[col] = pd.to_numeric(result[col].astype(str).str.replace(",", ""), errors="coerce")
                    audit_log.append(_log(rec_id, action, col, "applied", f"Converted '{col}' to float."))

            elif action == "convert_to_datetime":
                if col and col in result.columns:
                    result[col] = pd.to_datetime(result[col], infer_datetime_format=True, errors="coerce")
                    audit_log.append(_log(rec_id, action, col, "applied", f"Converted '{col}' to datetime."))

            elif action == "standardize_datetime":
                if col and col in result.columns:
                    result[col] = pd.to_datetime(result[col], infer_datetime_format=True, errors="coerce")
                    result[col] = result[col].dt.strftime("%Y-%m-%d")
                    audit_log.append(_log(rec_id, action, col, "applied", f"Standardised '{col}' to YYYY-MM-DD format."))

            elif action == "convert_to_boolean":
                if col and col in result.columns:
                    true_vals = {"true", "yes", "1", "y", "t"}
                    result[col] = result[col].astype(str).str.lower().str.strip().map(
                        lambda x: True if x in true_vals else (False if x not in ("nan", "none", "") else pd.NA)
                    )
                    audit_log.append(_log(rec_id, action, col, "applied", f"Converted '{col}' to boolean."))

            elif action == "remove_currency_symbol_and_convert_numeric":
                if col and col in result.columns:
                    result[col] = pd.to_numeric(
                        result[col].astype(str).str.replace(r"[\$€£¥,\s]", "", regex=True),
                        errors="coerce",
                    )
                    audit_log.append(_log(rec_id, action, col, "applied", f"Removed currency symbols and converted '{col}' to numeric."))

            elif action == "remove_percent_symbol_and_convert_numeric":
                if col and col in result.columns:
                    result[col] = pd.to_numeric(
                        result[col].astype(str).str.replace("%", "", regex=False).str.strip(),
                        errors="coerce",
                    )
                    audit_log.append(_log(rec_id, action, col, "applied", f"Removed '%' symbols and converted '{col}' to numeric."))

            # ── String normalization ──────────────────────────────────────
            elif action == "strip_and_normalize":
                if col and col in result.columns:
                    result[col] = result[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.title()
                    result[col] = result[col].replace("Nan", pd.NA)
                    audit_log.append(_log(rec_id, action, col, "applied", f"Stripped whitespace and normalized case in '{col}'."))

            elif action == "to_lower":
                if col and col in result.columns:
                    result[col] = result[col].astype(str).str.lower().str.strip()
                    result[col] = result[col].replace("nan", pd.NA)
                    audit_log.append(_log(rec_id, action, col, "applied", f"Converted '{col}' to lowercase."))

            elif action == "to_upper":
                if col and col in result.columns:
                    result[col] = result[col].astype(str).str.upper().str.strip()
                    result[col] = result[col].replace("NAN", pd.NA)
                    audit_log.append(_log(rec_id, action, col, "applied", f"Converted '{col}' to uppercase."))

            elif action == "to_title_case":
                if col and col in result.columns:
                    result[col] = result[col].astype(str).str.title().str.strip()
                    result[col] = result[col].replace("Nan", pd.NA)
                    audit_log.append(_log(rec_id, action, col, "applied", f"Converted '{col}' to title case."))

            elif action == "standardize_categories":
                if col and col in result.columns:
                    result[col] = result[col].astype(str).str.strip().str.title()
                    result[col] = result[col].replace("Nan", pd.NA)
                    audit_log.append(_log(rec_id, action, col, "applied", f"Standardised categories in '{col}'."))

            # ── Outlier handling ──────────────────────────────────────────
            elif action == "remove_outliers":
                if col and col in result.columns:
                    q1 = result[col].quantile(0.25)
                    q3 = result[col].quantile(0.75)
                    iqr = q3 - q1
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    mask = (result[col] >= lower) & (result[col] <= upper) | result[col].isna()
                    removed = int((~mask).sum())
                    result = result[mask]
                    audit_log.append(_log(rec_id, action, col, "applied", f"Removed {removed} outlier rows from '{col}'."))

            elif action == "cap_winsorize":
                if col and col in result.columns:
                    q1 = result[col].quantile(0.25)
                    q3 = result[col].quantile(0.75)
                    iqr = q3 - q1
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    n_capped = int(((result[col] < lower) | (result[col] > upper)).sum())
                    result[col] = result[col].clip(lower=lower, upper=upper)
                    audit_log.append(_log(rec_id, action, col, "applied", f"Winsorized {n_capped} outliers in '{col}'."))

            elif action == "replace_with_median":
                if col and col in result.columns:
                    q1 = result[col].quantile(0.25)
                    q3 = result[col].quantile(0.75)
                    iqr = q3 - q1
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    median_val = result[col].median()
                    mask = (result[col] < lower) | (result[col] > upper)
                    n_replaced = int(mask.sum())
                    result.loc[mask, col] = median_val
                    audit_log.append(_log(rec_id, action, col, "applied", f"Replaced {n_replaced} outliers with median ({median_val:.4f}) in '{col}'."))

            # ── Invalid value handling ────────────────────────────────────
            elif action == "remove_invalid_rows":
                extra = op.get("extra_data", {})
                issue_type = extra.get("issue_type", "")
                if col and col in result.columns:
                    if issue_type == "invalid_age":
                        mask = (result[col] >= 0) & (result[col] <= 120) | result[col].isna()
                    elif issue_type == "invalid_percentage":
                        mask = ((result[col] >= 0) & (result[col] <= 100)) | result[col].isna()
                    else:
                        mask = pd.Series([True] * len(result), index=result.index)
                    removed = int((~mask).sum())
                    result = result[mask]
                    audit_log.append(_log(rec_id, action, col, "applied", f"Removed {removed} invalid rows from '{col}'."))

            elif action == "replace_with_nan":
                extra = op.get("extra_data", {})
                issue_type = extra.get("issue_type", "")
                if col and col in result.columns:
                    if issue_type == "invalid_age":
                        mask = (result[col] < 0) | (result[col] > 120)
                    else:
                        mask = pd.Series([False] * len(result), index=result.index)
                    n = int(mask.sum())
                    result.loc[mask, col] = pd.NA
                    audit_log.append(_log(rec_id, action, col, "applied", f"Set {n} invalid values to NaN in '{col}'."))

            else:
                audit_log.append(_log(rec_id, action, col, "skipped", f"Unknown action '{action}'."))

        except Exception as exc:
            audit_log.append(_log(rec_id, action, col, "error", str(exc)))

    result = result.reset_index(drop=True)
    return result, audit_log


def _log(rec_id: str, action: str, col: Optional[str], status: str, message: str) -> Dict[str, Any]:
    return {
        "recommendation_id": rec_id,
        "action": action,
        "column": col,
        "status": status,
        "message": message,
    }
