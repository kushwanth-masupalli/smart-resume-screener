# Smart Resume Screener — Backend

A production-grade resume screening API built with FastAPI, PostgreSQL + pgvector, and a cascading AI pipeline that reduces LLM costs by 80–90%.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Pipeline Deep Dive](#pipeline-deep-dive)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Database Schema](#database-schema)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Testing](#testing)

---

## Architecture Overview

```
                         ┌─────────────────────────────────────────┐
                         │           UPLOAD / INGEST                │
                         │  PDF/TXT → text extraction → SHA-256    │
                         └────────────────────┬────────────────────┘
                                              │
                         ┌────────────────────▼────────────────────┐
                         │         STRUCTURING (spaCy + regex)      │
                         │  Name · Email · Phone · Skills · Edu     │
                         └────────────────────┬────────────────────┘
                                              │
                         ┌────────────────────▼────────────────────┐
                         │         EMBEDDING (sentence-transformers)│
                         │  384-dim vectors via all-MiniLM-L6-v2    │
                         └────────────────────┬────────────────────┘
                                              │
        ┌─────────────────────────────────────▼─────────────────────────────────────┐
        │                          SCREENING PIPELINE                              │
        │                                                                          │
        │  ┌─────────────┐     ┌─────────────┐     ┌──────────────┐     ┌────────┐│
        │  │  THE SIEVE   │────▶│  THE JUDGE   │────▶│ GROUNDING    │────▶│ GAP    ││
        │  │  (embeddings)│     │  (LLM)       │     │ GUARD        │     │ COACH  ││
        │  │  Zero cost   │     │  Top-K only  │     │ Fact-check   │     │ (LLM)  ││
        │  └─────────────┘     └─────────────┘     └──────────────┘     └────────┘│
        │     N → K               K LLM calls         Verify claims     3 questions│
        │     (80-90% cut)                                                per shortlist│
        └──────────────────────────────────────────────────────────────────────────┘
```

### The Efficiency Story

| Stage | What it does | LLM calls | Cost |
|-------|-------------|-----------|------|
| **The Sieve** | Embeds JD once, embeds all resumes, ranks by cosine similarity | **0** | Free |
| **The Judge** | Structured JSON scoring of top-K survivors only | **K** (not N) | 80–90% less |
| **Grounding Guard** | Verifies LLM claims against extracted data | 0 | Free |
| **Blind Mode** | Strips PII before scoring | 0 | Free |
| **Gap Coach** | Generates 3 targeted interview questions | **K** | Minimal |

At **N=200 candidates, K=20 survivors**, that's a **90% reduction** in LLM API calls.

---

## Pipeline Deep Dive

### Stage 1: The Sieve

The Sieve is the efficiency core. It operates entirely locally with zero API cost.

1. **Embed the Job Description** — one call to `sentence-transformers`
2. **Embed all candidate summaries** — batched locally, ~10ms per 100 candidates
3. **Cosine similarity ranking** — instant vector math via numpy
4. **Select top-K** — configurable threshold (default 20%), min 5, max 50

```python
# From app/services/sieve.py
sieve_result = sieve(
    jd_text=job.raw_text,
    candidate_summaries=summaries,
    top_k_percent=0.20,   # Pass top 20% to Judge
    min_candidates=5,      # At least 5 candidates
    max_candidates=50,     # At most 50 candidates
)
```

### Stage 2: The Judge

The Judge is the only component that calls the LLM. It uses **forced structured JSON output** for consistency and auditability.

- **Model**: `meta/llama-3.3-70b-instruct` via NVIDIA NIM API
- **Temperature**: 0.3 (deterministic, calibrated scoring)
- **Schema**: Forced JSON with `skills_match`, `experience_match`, `education_match`, `overall_score` (each 0–10), plus `matched_skills`, `missing_skills`, `justification`, and `red_flags`
- **Retry logic**: 3 attempts with exponential backoff

```json
{
  "skills_match": 7,
  "experience_match": 6,
  "education_match": 8,
  "overall_score": 7,
  "matched_skills": ["python", "fastapi", "postgresql"],
  "missing_skills": ["kubernetes", "terraform"],
  "justification": "Strong backend experience with Python and FastAPI at Acme Corp...",
  "red_flags": []
}
```

### Grounding Guard

Post-Judge verification that catches LLM hallucinations.

- Cross-checks each `matched_skills` entry against the extracted skill list
- Uses fuzzy matching (`SequenceMatcher`) with a 0.6 similarity threshold
- Returns `GroundingFlag` entries with confidence levels:
  - **high** — skill claimed by LLM but not found in resume
  - **medium** — partial/fuzzy match detected

### Blind Mode

Bias mitigation through identity redaction before LLM scoring.

**Redacted elements:**
- Names (spaCy NER + regex)
- Emails, phone numbers, addresses
- Graduation years (age proxies)
- Pronouns, gender-specific titles
- LinkedIn/GitHub/website URLs
- Marital status, nationality indicators

The redacted text is scored by the Judge. Identity is only re-attached for display.

### Gap Coach

For shortlisted candidates only (low cost — runs on top-K only).

- Analyzes the candidate's weak areas from Judge output
- Generates **3 targeted interview questions** mixing behavioral and technical
- Focuses on missing skills and low-scoring dimensions
- Falls back to generic questions if LLM call fails

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | FastAPI 0.109 |
| **Database** | PostgreSQL 15+ with pgvector extension |
| **ORM** | SQLAlchemy 2.0 (async) |
| **PDF Parsing** | pdfplumber + PyMuPDF |
| **NLP** | spaCy 3.7 (`en_core_web_sm`) |
| **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) |
| **LLM** | NVIDIA NIM API (OpenAI-compatible) — Llama 3.3 70B |
| **Vector Search** | pgvector (cosine similarity) |
| **Export** | openpyxl (Excel) |
| **Container** | Docker (Python 3.11-slim) |

---

## Prerequisites

- **Python 3.11+**
- **PostgreSQL 15+** with `pgvector` extension
- **NVIDIA NIM API key** (free tier at https://build.nvidia.com)
- **~2GB disk** for spaCy model + sentence-transformers weights

---

## Quick Start

### 1. Clone and install

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 3. Set up PostgreSQL

```bash
# Create database
createdb resume_screener

# Enable pgvector extension
psql -d resume_screener -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/resume_screener
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxx
```

### 5. Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: **http://localhost:8000/docs**

---

## Configuration

All settings are loaded from environment variables (or `.env` file) via `pydantic-settings`.

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/resume_screener` | Async PostgreSQL connection string |
| `SYNC_DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/resume_screener` | Sync connection for Alembic migrations |
| `NVIDIA_API_KEY` | — | Your NVIDIA NIM API key |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | OpenAI-compatible API base URL |
| `LLM_MODEL` | `meta/llama-3.3-70b-instruct` | LLM model identifier |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `EMBEDDING_DIMENSION` | `384` | Vector dimension (must match model) |
| `SPACY_MODEL` | `en_core_web_sm` | spaCy NER model |
| `SIEVE_TOP_K_PERCENT` | `0.20` | Default % of candidates to pass to Judge |
| `SIEVE_MIN_CANDIDATES` | `5` | Minimum candidates sent to Judge |
| `SIEVE_MAX_CANDIDATES` | `50` | Maximum candidates sent to Judge |

---

## API Reference

Base URL: `http://localhost:8000/api/v1`

### Health Check

```
GET /api/v1/health
```

**Response:**
```json
{"status": "ok", "service": "Smart Resume Screener"}
```

---

### Resumes

#### Upload a Resume

```
POST /api/v1/resumes/upload
Content-Type: multipart/form-data
```

| Field | Type | Required |
|-------|------|----------|
| `file` | File (.pdf, .txt, .md) | Yes |

**Response:** `201 Created`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "John Doe",
  "email": "john@example.com",
  "resume_hash": "a1b2c3d4...",
  "created_at": "2025-01-15T10:30:00Z"
}
```

#### Upload Multiple Resumes

```
POST /api/v1/resumes/upload-batch
Content-Type: multipart/form-data
```

| Field | Type | Required |
|-------|------|----------|
| `files` | File[] | Yes |

#### List Resumes

```
GET /api/v1/resumes?skip=0&limit=100
```

#### Get Resume Detail

```
GET /api/v1/resumes/{candidate_id}
```

Returns full resume data including `raw_text`, `parsed_json`, and `resume_hash`.

#### Delete Resume

```
DELETE /api/v1/resumes/{candidate_id}
```

---

### Job Descriptions

#### Create Job

```
POST /api/v1/jobs
Content-Type: multipart/form-data
```

| Field | Type | Required |
|-------|------|----------|
| `title` | string | Yes |
| `text` | string | Yes |

**Response:** `201 Created`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440001",
  "title": "Senior Python Developer",
  "created_at": "2025-01-15T10:35:00Z"
}
```

#### List Jobs

```
GET /api/v1/jobs?skip=0&limit=100
```

#### Get Job Detail

```
GET /api/v1/jobs/{job_id}
```

#### Delete Job

```
DELETE /api/v1/jobs/{job_id}
```

---

### Screening

#### Run Screening Pipeline

```
POST /api/v1/screen/{job_id}
Content-Type: application/json
```

**Request Body:**
```json
{
  "blind_mode": false,
  "generate_gap_coach": true,
  "sieve_params": {
    "top_k_percent": 0.30,
    "min_candidates": 3,
    "max_candidates": 50
  }
}
```

**Response:** `200 OK`
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440001",
  "job_title": "Senior Python Developer",
  "total_candidates": 150,
  "sieve_survivors": 30,
  "llm_calls_made": 30,
  "cost_reduction_percent": 80.0,
  "matches": [...],
  "screening_time_seconds": 12.5,
  "created_at": "2025-01-15T10:35:00Z"
}
```

#### Get Cached Screening Results

```
GET /api/v1/screen/{job_id}/results?sort_by=overall_score&sort_order=desc&min_score=5&search=john
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sort_by` | string | `overall_score` | Sort field: `overall_score`, `skills_match`, `experience_match`, `education_match`, `sieve_score`, `sieve_rank` |
| `sort_order` | string | `desc` | `asc` or `desc` |
| `min_score` | float | — | Minimum overall score (0–10) |
| `max_score` | float | — | Maximum overall score (0–10) |
| `min_skills` | int | — | Minimum skills_match score |
| `min_experience` | int | — | Minimum experience_match score |
| `has_red_flags` | bool | — | Filter by red flags presence |
| `has_hallucinations` | bool | — | Filter by hallucination flags |
| `blind_mode` | bool | — | Filter by blind mode usage |
| `search` | string | — | Search candidate name (case-insensitive) |
| `skip` | int | `0` | Pagination offset |
| `limit` | int | `100` | Pagination limit (max 500) |

Results are tiebreaker-sorted: when two candidates have the same overall score, secondary criteria are `skills_match` → `experience_match` → `education_match` → `sieve_score` → `sieve_rank`.

---

### Excel Export

#### Export Screening Results

```
GET /api/v1/screen/{job_id}/export
```

Returns an `.xlsx` file with two sheets:

1. **Screening Results** — One row per candidate with columns:
   - Candidate Name, Email, Rejection Phase
   - Sieve Score, Sieve Rank
   - Overall Score, Skills Match, Experience Match, Education Match
   - Matched Skills, Missing Skills, Justification
   - Red Flags, Hallucination Flags, Blind Mode

2. **Summary** — Aggregate metrics: total candidates, sieve survivors, average score, cost reduction percentage

Phase indicators are color-coded:
- 🟢 **All Stages Passed** (green)
- 🟡 **Judge (LLM Scored)** (yellow)
- 🔴 **Sieve (Pre-filter)** (red — eliminated before LLM)

---

### Statistics

```
GET /api/v1/stats
```

**Response:**
```json
{
  "total_candidates": 150,
  "total_jobs": 12,
  "total_matches": 450,
  "llm_scored_matches": 90
}
```

---

## Database Schema

### `candidates` Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID (PK) | Unique identifier |
| `name` | VARCHAR(255) | Extracted name (spaCy NER) |
| `email` | VARCHAR(255) | Extracted email |
| `phone` | VARCHAR(100) | Extracted phone |
| `resume_hash` | VARCHAR(64) | SHA-256 hash (dedup key) |
| `raw_text` | TEXT | Full extracted text |
| `parsed_json` | JSONB | Structured extraction results |
| `embedding` | VECTOR(384) | Sentence-transformer vector |
| `created_at` | TIMESTAMP | Ingestion timestamp |
| `updated_at` | TIMESTAMP | Last update timestamp |

### `job_descriptions` Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID (PK) | Unique identifier |
| `title` | VARCHAR(255) | Job title |
| `raw_text` | TEXT | Full job description text |
| `parsed_json` | JSONB | Parsed structure (optional) |
| `embedding` | VECTOR(384) | Sentence-transformer vector |
| `created_at` | TIMESTAMP | Creation timestamp |

### `matches` Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID (PK) | Unique identifier |
| `candidate_id` | UUID (FK) | References `candidates.id` |
| `job_id` | UUID (FK) | References `job_descriptions.id` |
| `sieve_score` | FLOAT | Cosine similarity score |
| `sieve_rank` | INT | Rank within job's candidates |
| `judge_json` | JSONB | Full Judge output |
| `overall_score` | FLOAT | Extracted for sorting |
| `skills_match` | FLOAT | Judge sub-score |
| `experience_match` | FLOAT | Judge sub-score |
| `education_match` | FLOAT | Judge sub-score |
| `grounding_flags` | JSONB | Hallucination flags |
| `gap_coach_questions` | JSONB | Generated questions |
| `blind_mode` | INT | 0=normal, 1=blind |
| `processed_at` | TIMESTAMP | When screening ran |
| `created_at` | TIMESTAMP | Creation timestamp |

---

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app, CORS, lifespan, router setup
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py           # All API endpoint handlers
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # Settings from .env (pydantic-settings)
│   │   ├── database.py         # Async SQLAlchemy engine + session
│   │   └── embeddings.py       # sentence-transformers wrapper
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database_models.py  # SQLAlchemy ORM models
│   │   └── schemas.py          # Pydantic request/response schemas
│   └── services/
│       ├── __init__.py
│       ├── orchestrator.py     # Pipeline coordinator
│       ├── ingest.py           # PDF/text extraction + SHA-256
│       ├── structuring.py      # spaCy NER + regex extraction
│       ├── sieve.py            # Embedding pre-filter (Stage 1)
│       ├── judge.py            # LLM scoring (Stage 2)
│       ├── grounding_guard.py  # Hallucination detection
│       ├── blind_mode.py       # PII redaction
│       ├── gap_coach.py        # Interview question generation
│       └── export.py           # Excel export (openpyxl)
├── tests/
│   └── test_e2e.py             # End-to-end pipeline test
├── .env.example                # Environment variable template
├── Dockerfile                  # Docker build configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Deployment

### Docker

```bash
docker build -t smart-screener-backend .
docker run -p 8000:8000 \
  -e DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/resume_screener" \
  -e NVIDIA_API_KEY="nvapi-xxx" \
  smart-screener-backend
```

### Render (via render.yaml)

The project includes a `render.yaml` at the project root for one-click deployment to Render. It provisions:

- **Web Service** — FastAPI backend on port 8000
- **PostgreSQL Database** — with pgvector extension enabled

### Environment Variables (Production)

Set these in your deployment platform:

| Variable | Required | Notes |
|----------|----------|-------|
| `DATABASE_URL` | ✅ | Must use `postgresql+asyncpg://` scheme |
| `NVIDIA_API_KEY` | ✅ | Get from https://build.nvidia.com |
| `FRONTEND_URL` | Optional | CORS origin for production frontend URL |
| `LLM_MODEL` | Optional | Default: `meta/llama-3.3-70b-instruct` |
| `EMBEDDING_MODEL` | Optional | Default: `all-MiniLM-L6-v2` |

---

## Testing

```bash
cd backend
python -m pytest tests/ -v
```

The end-to-end test (`test_e2e.py`) validates the full pipeline: ingest → structure → embed → sieve → judge → grounding guard → gap coach.

---

## How It Works — Data Flow

1. **Upload Resume** → `ingest.py` extracts text (pdfplumber), computes SHA-256 hash
2. **Structure** → `structuring.py` runs spaCy NER for name/email/skills (no LLM)
3. **Embed** → `embeddings.py` generates 384-dim vector via `sentence-transformers`
4. **Store** → PostgreSQL + pgvector stores text, structured JSON, and embedding
5. **Screen** → `orchestrator.py` coordinates the full pipeline:
   - Sieve ranks all candidates by cosine similarity → top-K survivors
   - Judge calls LLM only on survivors → structured JSON scores
   - Grounding Guard verifies claims → hallucination flags
   - Blind Mode optionally strips PII before scoring
   - Gap Coach generates interview questions for shortlisted candidates
6. **Export** → `export.py` generates Excel workbook with screening results and summary metrics
7. **Results** → All scores, flags, and questions stored in `matches` table

---

## License

Internal project — see repository root for license details.
