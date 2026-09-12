"""
CleanSense AI — File Parser Service

Supports: CSV, XLSX, XLS, TXT, JSON, TSV
"""
import io
import json
import chardet
import pandas as pd
from typing import Optional, Tuple


# Extended set of values treated as missing
MISSING_REPRESENTATIONS = {
    "", " ", "na", "n/a", "null", "none", "nan", "?", "-", "missing",
    "N/A", "NA", "NULL", "None", "NaN", "MISSING",
}


def _standardise_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace all common missing-value representations with NaN."""
    return df.replace(list(MISSING_REPRESENTATIONS), pd.NA)


def _detect_delimiter(sample: str) -> str:
    """Heuristically detect the delimiter in a text file."""
    candidates = {",": 0, "\t": 0, ";": 0, "|": 0, " ": 0}
    for line in sample.splitlines()[:20]:
        for sep in candidates:
            candidates[sep] += line.count(sep)
    best = max(candidates, key=candidates.get)
    return best if candidates[best] > 0 else ","


def parse_file(
    file_bytes: bytes,
    filename: str,
    sheet_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Parse an uploaded file into a DataFrame.

    Returns
    -------
    df : pd.DataFrame
    meta : dict  — extra metadata (sheets list, delimiter, etc.)
    """
    ext = filename.rsplit(".", 1)[-1].lower()
    meta: dict = {"file_type": ext, "delimiter": None, "sheets": None}

    # ── Excel ──────────────────────────────────────────────────────────────
    if ext in ("xlsx", "xls"):
        try:
            xl = pd.ExcelFile(io.BytesIO(file_bytes))
            sheets = xl.sheet_names
            meta["sheets"] = sheets
            target = sheet_name or sheets[0]
            df = xl.parse(target)
        except Exception as e:
            raise ValueError(f"Could not read Excel file: {e}")

    # ── JSON ───────────────────────────────────────────────────────────────
    elif ext == "json":
        try:
            raw = json.loads(file_bytes.decode("utf-8", errors="replace"))
            if isinstance(raw, list):
                df = pd.DataFrame(raw)
            elif isinstance(raw, dict):
                df = pd.DataFrame([raw]) if not any(isinstance(v, list) for v in raw.values()) \
                    else pd.DataFrame(raw)
            else:
                raise ValueError("Unsupported JSON structure")
        except Exception as e:
            raise ValueError(f"Could not parse JSON file: {e}")

    # ── CSV / TXT / TSV ────────────────────────────────────────────────────
    elif ext in ("csv", "txt", "tsv"):
        # Detect encoding
        detected = chardet.detect(file_bytes[:50_000])
        encoding = detected.get("encoding") or "utf-8"

        text = file_bytes.decode(encoding, errors="replace")

        if ext == "tsv":
            delimiter = "\t"
        elif ext == "csv":
            delimiter = ","
        else:  # txt — auto-detect
            delimiter = _detect_delimiter(text)

        meta["delimiter"] = delimiter

        try:
            df = pd.read_csv(
                io.StringIO(text),
                sep=delimiter,
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as e:
            raise ValueError(f"Could not parse text file: {e}")

    else:
        raise ValueError(f"Unsupported file type: .{ext}")

    # Standardise column names (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]

    # Replace common missing-value representations
    df = _standardise_missing(df)

    return df, meta
