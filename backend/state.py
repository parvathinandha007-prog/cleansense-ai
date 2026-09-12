"""
CleanSense AI — Shared in-memory session store.

Keyed by session UUID.  Each value is a dict:
  {
    raw_bytes:       bytes,
    filename:        str,
    meta:            dict,
    df_original:     pd.DataFrame | None,
    df_cleaned:      pd.DataFrame | None,
    history:         list[pd.DataFrame],   # undo stack
    operations_log:  list[dict],
    recommendations: list[dict],           # cached
    quality_score:   int,
    score_breakdown: dict,
  }
"""

sessions: dict = {}
