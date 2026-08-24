# Smart Resume Screener

An AI-powered resume screening system that uses a **cascading pipeline** to reduce LLM costs by 80–90% while delivering explainable, auditable candidate rankings.

Most resume screeners shove the full resume + JD into an LLM and get back a number. This one **engineers around the cost of LLM calls** — local embeddings pre-filter candidates, and the LLM only scores the survivors.

---

## Architecture

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

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.11 · FastAPI · SQLAlchemy 2.0 (async) |
| **Database** | PostgreSQL 15+ with pgvector |
| **NLP** | spaCy 3.7 (`en_core_web_sm`) |
| **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) |
| **LLM** | NVIDIA NIM API — Llama 3.3 70B Instruct |
| **PDF Parsing** | pdfplumber + PyMuPDF |
| **Frontend** | Single-file HTML (vanilla JS + CSS) |
| **Export** | openpyxl (Excel) |
| **Deploy** | Render (web services + PostgreSQL) |

---

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ with pgvector
- NVIDIA NIM API key (free at https://build.nvidia.com)

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Set up your `.env`:

```bash
cp .env.example .env
# Edit .env with your DATABASE_URL and NVIDIA_API_KEY
```

Create the database and enable pgvector:

```bash
createdb resume_screener
psql -d resume_screener -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Run the server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: **http://localhost:8000/docs**

### 2. Frontend

```bash
cd frontend
python -m http.server 5173
```

Open **http://localhost:5173** in your browser. The frontend talks to the backend at `http://localhost:8000/api/v1`.

---

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app, CORS, lifespan
│   │   ├── api/routes.py           # All API endpoints
│   │   ├── core/
│   │   │   ├── config.py           # pydantic-settings config
│   │   │   ├── database.py         # Async SQLAlchemy engine
│   │   │   └── embeddings.py       # sentence-transformers wrapper
│   │   ├── models/
│   │   │   ├── database_models.py  # SQLAlchemy ORM models
│   │   │   └── schemas.py          # Pydantic request/response schemas
│   │   └── services/
│   │       ├── orchestrator.py     # Pipeline coordinator
│   │       ├── ingest.py           # PDF/text extraction + SHA-256
│   │       ├── structuring.py      # spaCy NER + regex extraction
│   │       ├── sieve.py            # Embedding pre-filter (Stage 1)
│   │       ├── judge.py            # LLM scoring (Stage 2)
│   │       ├── grounding_guard.py  # Hallucination detection
│   │       ├── blind_mode.py       # PII redaction
│   │       └── export.py           # Excel export
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   └── index.html                  # Single-file frontend (HTML + CSS + JS)
├── render.yaml                     # Render deployment config
└── README.md                       # This file
```

---

## Deployment

The project includes a `render.yaml` for one-click deployment to Render:

- **Backend** — FastAPI web service (Python 3.11)
- **Frontend** — Static site (single HTML file, no build step)
- **Database** — PostgreSQL with pgvector (starter plan)

Set the `NVIDIA_API_KEY` in Render's environment variables after provisioning.

---

## How It Works

1. **Upload Resumes** — PDFs or text files are parsed, SHA-256 hashed (dedup), structured via spaCy, and embedded into 384-dim vectors
2. **Create a Job Description** — Paste or type the JD; it gets embedded the same way
3. **Run Screening** — The Sieve ranks all candidates by cosine similarity → top-K survivors go to the Judge → LLM scores survivors with forced JSON output → Grounding Guard catches hallucinations → optionally generates interview questions
4. **View Results** — Sorted by overall score with sub-scores, skill matching, red flags, and hallucination warnings. Export to Excel.

---

## License

Internal project — see repository root for license details.
