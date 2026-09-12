"""
CleanSense AI — Download Router
"""
import io
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse

from services.report_generator import generate_cleaning_report
from services.scorer import calculate_data_quality_score
from state import sessions

router = APIRouter(prefix="/api", tags=["download"])


@router.get("/download/{session_id}/csv")
async def download_csv(session_id: str, dataset: str = "cleaned"):
    session = _get_session(session_id)
    df = session["df_cleaned"] if dataset == "cleaned" else session["df_original"]

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    filename = _base_filename(session, "cleaned") + ".csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/download/{session_id}/excel")
async def download_excel(session_id: str, dataset: str = "cleaned"):
    session = _get_session(session_id)
    df = session["df_cleaned"] if dataset == "cleaned" else session["df_original"]

    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)

    filename = _base_filename(session, "cleaned") + ".xlsx"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/download/{session_id}/txt")
async def download_txt(session_id: str, dataset: str = "cleaned"):
    session = _get_session(session_id)
    df = session["df_cleaned"] if dataset == "cleaned" else session["df_original"]

    buf = io.StringIO()
    df.to_csv(buf, index=False, sep="\t")
    buf.seek(0)

    filename = _base_filename(session, "cleaned") + ".txt"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/download/{session_id}/report")
async def download_report(session_id: str):
    session = _get_session(session_id)
    orig = session["df_original"]
    cleaned = session["df_cleaned"]
    ops = session.get("operations_log", [])
    score_before = session.get("quality_score", 0)
    score_after, _ = calculate_data_quality_score(cleaned)

    report_md = generate_cleaning_report(orig, cleaned, ops, score_before, score_after)

    filename = _base_filename(session, "report") + ".md"
    return PlainTextResponse(
        content=report_md,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_session(session_id: str) -> dict:
    session = sessions.get(session_id)
    if not session or session.get("df_original") is None:
        raise HTTPException(404, "Session not found.")
    return session


def _base_filename(session: dict, suffix: str) -> str:
    raw = session.get("filename", "dataset")
    stem = raw.rsplit(".", 1)[0] if "." in raw else raw
    return f"{stem}_{suffix}"
