"""
CleanSense AI — Pydantic Schemas
"""
from pydantic import BaseModel
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Upload / Session
# ---------------------------------------------------------------------------

class SheetInfo(BaseModel):
    sheets: List[str]
    session_id: str


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    rows: int
    columns: int
    column_names: List[str]
    preview: List[Dict[str, Any]]          # first 10 rows
    delimiter_detected: Optional[str] = None
    sheets: Optional[List[str]] = None     # Excel multi-sheet


class SheetSelectRequest(BaseModel):
    session_id: str
    sheet_name: str


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

class ColumnProfile(BaseModel):
    name: str
    dtype: str
    category: str            # numerical / categorical / boolean / datetime / text
    non_null_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    cardinality: str         # low / medium / high
    is_constant: bool
    is_near_constant: bool
    sample_values: List[Any]
    # numerical extras
    mean: Optional[float] = None
    median: Optional[float] = None
    mode: Optional[Any] = None
    std: Optional[float] = None
    variance: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    iqr: Optional[float] = None
    skewness: Optional[float] = None
    outlier_count: Optional[int] = None
    outlier_percentage: Optional[float] = None


class AnalysisResponse(BaseModel):
    session_id: str
    rows: int
    columns: int
    numerical_cols: List[str]
    categorical_cols: List[str]
    boolean_cols: List[str]
    datetime_cols: List[str]
    text_cols: List[str]
    column_profiles: List[ColumnProfile]
    duplicate_rows: int
    duplicate_percentage: float
    total_missing: int
    total_missing_percentage: float
    quality_score: int
    score_breakdown: Dict[str, float]
    issues_detected: int
    issues_by_severity: Dict[str, int]


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

class Recommendation(BaseModel):
    id: str
    column: Optional[str]
    issue_type: str
    severity: str            # critical / high / medium / low
    title: str
    description: str
    count: Optional[int] = None
    percentage: Optional[float] = None
    recommendation: str
    reason: str
    possible_actions: List[str]
    default_action: str
    requires_user_approval: bool = True
    extra_data: Optional[Dict[str, Any]] = None


class RecommendationsResponse(BaseModel):
    session_id: str
    recommendations: List[Recommendation]
    total: int
    by_severity: Dict[str, int]


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

class CleaningOperation(BaseModel):
    recommendation_id: str
    action: str              # e.g. "median_imputation", "drop_duplicates", …
    column: Optional[str] = None
    custom_value: Optional[Any] = None
    keep: Optional[str] = None     # first / last (for duplicates)
    subset_columns: Optional[List[str]] = None


class CleaningRequest(BaseModel):
    session_id: str
    operations: List[CleaningOperation]


class CleaningOperationResult(BaseModel):
    recommendation_id: str
    action: str
    column: Optional[str]
    status: str              # applied / skipped / error
    message: str
    rows_affected: Optional[int] = None


class BeforeAfterStats(BaseModel):
    rows_before: int
    rows_after: int
    missing_before: int
    missing_after: int
    duplicates_before: int
    duplicates_after: int
    quality_score_before: int
    quality_score_after: int


class CleaningResponse(BaseModel):
    session_id: str
    operations_applied: List[CleaningOperationResult]
    before_after: BeforeAfterStats
    history: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

class PreviewRequest(BaseModel):
    session_id: str
    page: int = 1
    page_size: int = 50
    search: Optional[str] = None
    sort_column: Optional[str] = None
    sort_direction: Optional[str] = "asc"
    dataset: str = "cleaned"   # original / cleaned


class PreviewResponse(BaseModel):
    session_id: str
    page: int
    page_size: int
    total_rows: int
    total_pages: int
    columns: List[str]
    data: List[Dict[str, Any]]
    problematic_cells: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------

class UndoResponse(BaseModel):
    session_id: str
    undone_operation: str
    history: List[Dict[str, Any]]
    before_after: BeforeAfterStats
