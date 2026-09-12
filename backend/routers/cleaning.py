"""
CleanSense AI — Cleaning Router
"""
from fastapi import APIRouter, HTTPException

from models.schemas import CleaningRequest, CleaningResponse, UndoResponse
from services.cleaner import apply_cleaning_operations
from services.scorer import calculate_data_quality_score
from services.detector import detect_duplicates
from state import sessions

router = APIRouter(prefix="/api", tags=["cleaning"])


@router.post("/clean/{session_id}", response_model=CleaningResponse)
async def apply_cleaning(session_id: str, req: CleaningRequest):
    """Apply user-approved cleaning operations to the dataset."""
    session = sessions.get(session_id)
    if not session or session.get("df_cleaned") is None:
        raise HTTPException(404, "Session not found.")

    df_before = session["df_cleaned"]

    # Snapshot for undo
    session["history"].append(df_before.copy())

    # Convert Pydantic ops to dicts
    ops = [op.model_dump() for op in req.operations]

    df_after, audit_log = apply_cleaning_operations(df_before, ops)
    session["df_cleaned"] = df_after
    session.setdefault("operations_log", []).extend(audit_log)

    # Before stats
    score_before = session.get("quality_score", 0)
    missing_before = int(df_before.isna().sum().sum())
    dups_before = detect_duplicates(df_before)["duplicate_rows"]

    # After stats
    score_after, breakdown_after = calculate_data_quality_score(df_after)
    missing_after = int(df_after.isna().sum().sum())
    dups_after = detect_duplicates(df_after)["duplicate_rows"]

    session["quality_score"] = score_after
    session["score_breakdown"] = breakdown_after

    from models.schemas import BeforeAfterStats, CleaningOperationResult

    before_after = BeforeAfterStats(
        rows_before=len(df_before),
        rows_after=len(df_after),
        missing_before=missing_before,
        missing_after=missing_after,
        duplicates_before=dups_before,
        duplicates_after=dups_after,
        quality_score_before=score_before,
        quality_score_after=score_after,
    )

    results = [
        CleaningOperationResult(
            recommendation_id=entry["recommendation_id"],
            action=entry["action"],
            column=entry.get("column"),
            status=entry["status"],
            message=entry["message"],
        )
        for entry in audit_log
    ]

    history_summary = [
        {"step": i + 1, "description": f"{e['action']} on {e.get('column') or 'dataset'}"}
        for i, e in enumerate(session["operations_log"])
        if e["status"] == "applied"
    ]

    return CleaningResponse(
        session_id=session_id,
        operations_applied=results,
        before_after=before_after,
        history=history_summary,
    )


@router.post("/undo/{session_id}", response_model=UndoResponse)
async def undo_last_operation(session_id: str):
    """Undo the last cleaning operation by restoring a snapshot."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found.")

    history = session.get("history", [])
    if not history:
        raise HTTPException(400, "Nothing to undo.")

    prev_df = history.pop()
    session["df_cleaned"] = prev_df

    # Remove last applied op from log
    ops_log = session.get("operations_log", [])
    undone_desc = "Unknown"
    for i in range(len(ops_log) - 1, -1, -1):
        if ops_log[i]["status"] == "applied":
            undone_desc = f"{ops_log[i]['action']} on {ops_log[i].get('column') or 'dataset'}"
            ops_log.pop(i)
            break

    score_after, breakdown_after = calculate_data_quality_score(prev_df)
    missing_after = int(prev_df.isna().sum().sum())
    dups_after = detect_duplicates(prev_df)["duplicate_rows"]

    orig = session["df_original"]
    missing_orig = int(orig.isna().sum().sum())
    dups_orig = detect_duplicates(orig)["duplicate_rows"]
    score_orig = session.get("quality_score", 0)

    from models.schemas import BeforeAfterStats

    before_after = BeforeAfterStats(
        rows_before=len(orig),
        rows_after=len(prev_df),
        missing_before=missing_orig,
        missing_after=missing_after,
        duplicates_before=dups_orig,
        duplicates_after=dups_after,
        quality_score_before=score_orig,
        quality_score_after=score_after,
    )

    history_summary = [
        {"step": i + 1, "description": f"{e['action']} on {e.get('column') or 'dataset'}"}
        for i, e in enumerate(ops_log)
        if e["status"] == "applied"
    ]

    return UndoResponse(
        session_id=session_id,
        undone_operation=undone_desc,
        history=history_summary,
        before_after=before_after,
    )
