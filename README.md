# CleanSense AI 🧹

**An intelligent, AI-powered data cleaning assistant built with Python + FastAPI + Pandas.**

> Upload → Analyze → Detect Problems → Explain → Recommend → User Approves → Clean → Download

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📤 Multi-format Upload | CSV, XLSX, XLS, TXT, TSV, JSON with auto-delimiter detection |
| 🔬 Comprehensive Analysis | Column profiling, data types, statistics, cardinality |
| 🐛 12+ Issue Detectors | Missing values, duplicates, outliers, type mismatches, string issues, invalid values, dates, correlations |
| 💡 Smart Recommendations | Rule-based engine with severity levels and plain-English explanations |
| ✅ User Approval System | Accept/reject/customize every recommendation individually |
| ⚙️ Cleaning Pipeline | Ordered, safe execution with audit log |
| ↩️ Undo Support | DataFrame snapshots for reversible operations |
| 📊 Data Preview | Paginated spreadsheet-style view |
| 📈 Quality Scoring | 0–100 score with transparent penalty breakdown |
| 💾 Multi-format Export | CSV, Excel, TXT + Markdown cleaning report |
| 🔒 Privacy First | No permanent storage, no external AI API calls |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the Server

**Windows (double-click):**
```
start.bat
```

**Or manually:**
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open the Application

Navigate to: **http://localhost:8000**

API documentation: **http://localhost:8000/docs**

---

## 🏗️ Architecture

```
cleansense-ai/
├── backend/
│   ├── main.py                  # FastAPI entry point
│   ├── state.py                 # In-memory session store
│   ├── requirements.txt
│   ├── static/
│   │   └── index.html           # Complete single-page frontend
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response models
│   ├── routers/
│   │   ├── upload.py            # File upload + sheet selection
│   │   ├── analysis.py          # Analysis + recommendations + preview
│   │   ├── cleaning.py          # Apply operations + undo
│   │   └── download.py          # CSV / Excel / TXT / report download
│   └── services/
│       ├── file_parser.py       # Multi-format file parsing
│       ├── detector.py          # 12+ modular issue detectors
│       ├── recommender.py       # Rule-based recommendation engine
│       ├── cleaner.py           # Cleaning operations executor
│       ├── scorer.py            # Data quality scoring
│       └── report_generator.py # Markdown report generation
└── start.bat                    # Windows startup script
```

---

## 🔬 Issue Detectors

| Detector | What it finds |
|----------|--------------|
| `detect_missing_values` | NaN, None, empty strings, "NA", "null", "?", "-", "missing" |
| `detect_duplicates` | Exact duplicate rows, near-duplicate key columns |
| `detect_outliers` | IQR + Modified Z-score methods |
| `detect_data_type_issues` | Numeric/date/boolean stored as strings, currency, percentages |
| `detect_string_inconsistencies` | Leading/trailing spaces, mixed case, extra spaces |
| `detect_constant_columns` | Constant and near-constant (≥98%) columns |
| `detect_high_missing_columns` | Columns >30% / >50% / >80% missing |
| `detect_invalid_values` | Age out of range, malformed emails, invalid percentages |
| `detect_date_issues` | Multiple formats, unparseable dates, future dates |
| `detect_high_cardinality` | Categorical columns with >50 unique values |
| `detect_categorical_inconsistencies` | Same category with different spellings/case |
| `detect_correlation` | Highly correlated column pairs (r ≥ 0.9) |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload file, create session |
| POST | `/api/upload/select-sheet` | Select Excel sheet |
| GET | `/api/analysis/{session_id}` | Full dataset analysis |
| GET | `/api/recommendations/{session_id}` | Recommendations |
| GET | `/api/preview/{session_id}` | Paginated data preview |
| POST | `/api/clean/{session_id}` | Apply cleaning operations |
| POST | `/api/undo/{session_id}` | Undo last operation |
| GET | `/api/download/{session_id}/csv` | Download as CSV |
| GET | `/api/download/{session_id}/excel` | Download as Excel |
| GET | `/api/download/{session_id}/txt` | Download as TXT |
| GET | `/api/download/{session_id}/report` | Download cleaning report |

---

## 🔐 Privacy & Security

- Files are stored **in-memory only** and deleted when the server restarts
- **No data is sent to external APIs** — all analysis uses Pandas/NumPy/SciPy
- Each session has a **UUID key** — no cross-session data access
- **50 MB file size limit** enforced

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10, FastAPI, Uvicorn |
| Data Processing | Pandas, NumPy, SciPy |
| File Parsing | openpyxl, xlrd, chardet |
| Frontend | Vanilla HTML/CSS/JS (no Node.js required) |
| Charts | Chart.js (CDN) |
| Fonts | Inter, JetBrains Mono (Google Fonts) |
