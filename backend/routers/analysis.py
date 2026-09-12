"""
CleanSense AI — Analysis Router
"""
from fastapi import APIRouter, HTTPException

from models.schemas import AnalysisResponse, RecommendationsResponse
from services.detector import (
    build_column_profile,
    detect_duplicates,
    detect_missing_values,
    _classify_column,
)
from services.recommender import generate_recommendations
from services.scorer import calculate_data_quality_score
from state import sessions

router = APIRouter(prefix="/api", tags=["analysis"])


@router.get("/analysis/{session_id}", response_model=AnalysisResponse)
async def get_analysis(session_id: str):
    """Return comprehensive dataset analysis for a session."""
    session = sessions.get(session_id)
    if not session or session.get("df_original") is None:
        raise HTTPException(404, "Session not found or file not yet loaded.")

    df = session["df_original"]

    # Column classification
    numerical_cols = list(df.select_dtypes(include=["number"]).columns)
    boolean_cols = [c for c in df.columns if df[c].dtype == bool]
    import pandas as pd
    datetime_cols = list(df.select_dtypes(include=["datetime64"]).columns)

    # Classify remaining object columns
    categorical_cols = []
    text_cols = []
    for col in df.select_dtypes(include=["object"]).columns:
        ct = _classify_column(df[col])
        if ct == "categorical":
            categorical_cols.append(col)
        elif ct == "datetime":
            datetime_cols.append(col)
        else:
            text_cols.append(col)

    # Column profiles
    profiles = build_column_profile(df)

    # Duplicates
    dup_info = detect_duplicates(df)

    # Missing
    total_missing = int(df.isna().sum().sum())
    total_missing_pct = round(total_missing / max(df.size, 1) * 100, 2)

    # Quality score
    score, breakdown = calculate_data_quality_score(df)

    # Recommendations for issue count
    recs = generate_recommendations(df)
    by_severity: dict = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for r in recs:
        by_severity[r["severity"]] = by_severity.get(r["severity"], 0) + 1

    # Cache recommendations in session
    session["recommendations"] = recs
    session["quality_score"] = score
    session["score_breakdown"] = breakdown

    return AnalysisResponse(
        session_id=session_id,
        rows=len(df),
        columns=len(df.columns),
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        boolean_cols=boolean_cols,
        datetime_cols=datetime_cols,
        text_cols=text_cols,
        column_profiles=profiles,
        duplicate_rows=dup_info["duplicate_rows"],
        duplicate_percentage=dup_info["percentage"],
        total_missing=total_missing,
        total_missing_percentage=total_missing_pct,
        quality_score=score,
        score_breakdown=breakdown,
        issues_detected=len(recs),
        issues_by_severity=by_severity,
    )


@router.get("/recommendations/{session_id}", response_model=RecommendationsResponse)
async def get_recommendations(session_id: str):
    """Return cached recommendations or generate them fresh."""
    session = sessions.get(session_id)
    if not session or session.get("df_original") is None:
        raise HTTPException(404, "Session not found.")

    if "recommendations" not in session:
        df = session["df_original"]
        recs = generate_recommendations(df)
        session["recommendations"] = recs

    recs = session["recommendations"]
    by_severity: dict = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for r in recs:
        by_severity[r["severity"]] = by_severity.get(r["severity"], 0) + 1

    return RecommendationsResponse(
        session_id=session_id,
        recommendations=recs,
        total=len(recs),
        by_severity=by_severity,
    )


@router.get("/preview/{session_id}")
async def get_preview(
    session_id: str,
    page: int = 1,
    page_size: int = 50,
    dataset: str = "cleaned",
):
    """Return a paginated preview of the dataset."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found.")

    df = session.get("df_cleaned") if dataset == "cleaned" else session.get("df_original")
    if df is None:
        raise HTTPException(404, "Dataset not found.")

    total_rows = len(df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    start = (page - 1) * page_size
    end = start + page_size

    page_df = df.iloc[start:end].copy()
    # Convert to serialisable
    data = page_df.fillna("").astype(str).to_dict(orient="records")

    return {
        "session_id": session_id,
        "page": page,
        "page_size": page_size,
        "total_rows": total_rows,
        "total_pages": total_pages,
        "columns": list(df.columns),
        "data": data,
    }
