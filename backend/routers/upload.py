"""
CleanSense AI — Upload Router
"""
import io
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from models.schemas import UploadResponse, SheetSelectRequest
from services.file_parser import parse_file
from state import sessions

router = APIRouter(prefix="/api", tags=["upload"])

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Form(None),
):
    """Upload a dataset file and create a new analysis session."""
    contents = await file.read()
    file_size = len(contents)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large. Maximum size is 50 MB.")
    if file_size == 0:
        raise HTTPException(400, "File is empty.")

    filename = file.filename or "upload"

    try:
        df, meta = parse_file(contents, filename, sheet_name=sheet_name)
    except ValueError as e:
        raise HTTPException(400, str(e))

    session_id = str(uuid.uuid4())

    # If Excel has multiple sheets and no sheet chosen, prompt for selection
    sheets = meta.get("sheets")
    if sheets and len(sheets) > 1 and not sheet_name:
        # Store partial session so user can pick sheet
        sessions[session_id] = {
            "raw_bytes": contents,
            "filename": filename,
            "meta": meta,
            "df_original": None,
            "df_cleaned": None,
            "history": [],
        }
        return UploadResponse(
            session_id=session_id,
            filename=filename,
            file_type=meta["file_type"],
            file_size_bytes=file_size,
            rows=0,
            columns=0,
            column_names=[],
            preview=[],
            delimiter_detected=meta.get("delimiter"),
            sheets=sheets,
        )

    sessions[session_id] = {
        "raw_bytes": contents,
        "filename": filename,
        "meta": meta,
        "df_original": df,
        "df_cleaned": df.copy(),
        "history": [],       # list of df snapshots for undo
        "operations_log": [],
    }

    preview = df.head(10).fillna("").astype(str).to_dict(orient="records")

    return UploadResponse(
        session_id=session_id,
        filename=filename,
        file_type=meta["file_type"],
        file_size_bytes=file_size,
        rows=len(df),
        columns=len(df.columns),
        column_names=list(df.columns),
        preview=preview,
        delimiter_detected=meta.get("delimiter"),
        sheets=sheets,
    )


@router.post("/upload/select-sheet", response_model=UploadResponse)
async def select_sheet(req: SheetSelectRequest):
    """Select a sheet for an Excel file after initial upload."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found.")

    try:
        df, meta = parse_file(
            session["raw_bytes"],
            session["filename"],
            sheet_name=req.sheet_name,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    session["df_original"] = df
    session["df_cleaned"] = df.copy()
    session["meta"] = meta
    session["history"] = []
    session["operations_log"] = []

    preview = df.head(10).fillna("").astype(str).to_dict(orient="records")
    file_size = len(session["raw_bytes"])

    return UploadResponse(
        session_id=req.session_id,
        filename=session["filename"],
        file_type=meta["file_type"],
        file_size_bytes=file_size,
        rows=len(df),
        columns=len(df.columns),
        column_names=list(df.columns),
        preview=preview,
        delimiter_detected=meta.get("delimiter"),
        sheets=meta.get("sheets"),
    )
