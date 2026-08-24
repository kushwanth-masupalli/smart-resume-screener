# 🚀 Smart Resume Screener

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Llama 3.3 70B](https://img.shields.io/badge/LLM-Llama_3.3_70B_Instruct-purple.svg)](https://build.nvidia.com/)
[![PostgreSQL pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1.svg)](https://github.com/pgvector/pgvector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, **cascading AI pipeline** designed for low-cost, hallucination-guarded, and auditable resume screening.

> **Submitted by**: Masupalli Kushwanth  
> **Email**: [masupallikushwanth2005@gmail.com](mailto:masupallikushwanth2005@gmail.com)  
> **GitHub**: [@kushwanth-masupalli](https://github.com/kushwanth-masupalli)

---

## 📌 Executive Summary & Problem Statement

Most AI-based HR tools suffer from two major production flaws:
1. **High Token Costs**: Naively feeding 200+ multi-page resumes directly into a high-parameter LLM (like GPT-4 or Llama-3.3 70B) burns millions of tokens, costing **$10–$50 per job posting** with high latency.
2. **Hallucinations & Bias**: Unconstrained LLMs frequently hallucinate candidate credentials or exhibit demographic bias based on names and locations.

### 💡 The Solution
**Smart Resume Screener** solves this by engineering around the cost of LLM calls using a **2-stage cascading architecture**:
* **Stage 1 (The Sieve)**: Computes 384-dimensional local embeddings via `sentence-transformers` to rank all candidates locally at **$0.00 LLM cost**.
* **Stage 2 (The Judge)**: Only sends the **top 10–20% survivors** (Sieve shortlist) to Llama 3.3 70B for deep structured JSON scoring.
* **Grounding Guard & Blind Mode**: Fact-checks LLM justifications against extracted resume facts and redacts PII before scoring.

**Result**: Achieves an **80–90% reduction in LLM API token consumption** while ensuring objective, auditable candidate rankings.

---

## 🏗️ Cascading Architecture Pipeline

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         INPUT & INGESTION                               │
  │  PDF / Text Resumes ──▶ SHA-256 Deduplication ──▶ Raw Text Extraction   │
  └───────────────────────────────────┬─────────────────────────────────────┘
                                      │
  ┌───────────────────────────────────▼─────────────────────────────────────┐
  │                         STRUCTURING (spaCy)                             │
  │  Entity Extraction: Name · Email · Phone · Hard Skills · Education      │
  └───────────────────────────────────┬─────────────────────────────────────┘
                                      │
  ┌───────────────────────────────────▼─────────────────────────────────────┐
  │               STAGE 1: THE SIEVE (Local Vector Search)                 │
  │  Compute 384-dim Embeddings (`all-MiniLM-L6-v2`)                        │
  │  Rank candidates by Cosine Similarity vs. Job Description               │
  │  ⚡ ZERO LLM COST — Filter N candidates down to Top-K survivors         │
  └───────────────────────────────────┬─────────────────────────────────────┘
                                      │
  ┌───────────────────────────────────▼─────────────────────────────────────┐
  │               STAGE 2: THE JUDGE (NVIDIA NIM Llama 3.3 70B)             │
  │  Deep Structured JSON Scoring (Skills, Experience, Education)           │
  │  Optional PII Redaction (Blind Mode)                                    │
  └───────────────────────────────────┬─────────────────────────────────────┘
                                      │
  ┌───────────────────────────────────▼─────────────────────────────────────┐
  │               STAGE 3: GROUNDING GUARD & GAP COACH                       │
  │  🛡️ Grounding Guard: Detects LLM hallucinations against spaCy facts     │
  │  💬 Gap Coach: Generates 3 targeted interview questions per finalist    │
  └───────────────────────────────────┬─────────────────────────────────────┘
                                      │
  ┌───────────────────────────────────▼─────────────────────────────────────┐
  │                       OUTPUT & REPORTING                                │
  │  Interactive Web UI Dashboard · REST API · Styled Excel Export (.xlsx)  │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Efficiency & Cost Benchmark

Assuming a hiring batch of **N = 200 candidates** for 1 job description:

| Pipeline Stage | Processing Method | LLM API Calls | Execution Cost | Latency per Candidate |
| :--- | :--- | :---: | :---: | :---: |
| **Naive LLM Approach** | Send all 200 resumes to Llama-70B | **200** | ~ \$10.00 – \$15.00 | ~ 3.5 sec / candidate |
| **Stage 1: The Sieve** | `all-MiniLM-L6-v2` Local Vector Cosine | **0** | **\$0.00** | **< 15 ms** |
| **Stage 2: The Judge** | Llama 3.3 70B on Top-20 survivors | **20** | ~ \$1.00 – \$1.50 | ~ 1.2 sec / survivor |
| **Stage 3: Gap Coach** | Llama 3.3 70B interview probe | **20** | ~ \$0.50 | ~ 0.8 sec / survivor |
| **Total Smart Pipeline** | **Cascading Filter System** | **40** | **~ 90% Cost Cut** | **~ 85% Faster** |

---

## 🔥 Key Features

- **🎯 The Sieve (Vector Pre-Filter)**: Leverages lightweight PyTorch transformer models to instantly score resume-to-JD vector alignment locally.
- **⚖️ The LLM Judge**: Evaluates candidates against job requirements returning strict Pydantic JSON with sub-scores (`skills_match`, `experience_match`, `education_match`) and detailed rationales.
- **🛡️ Grounding Guard (Anti-Hallucination)**: Scans LLM output against spaCy extracted entities to catch hallucinated years of experience, unlisted tools, or fabricated degrees.
- **🙈 Blind Mode (Bias Mitigation)**: Dynamically strips candidate names, emails, phone numbers, and location identifiers prior to LLM submission for fair evaluation.
- **💬 Gap Coach**: Automatically formulates 3 technical interview probing questions focused on candidate weak spots.
- **📊 Tiebreaker Matrix**: Resolves identical overall scores by evaluating secondary criteria: `skills_match` ➔ `experience_match` ➔ `education_match` ➔ `sieve_vector_rank`.
- **📥 One-Click Excel Export**: Generates beautifully styled, multi-tab Excel reports (`openpyxl`) formatted with color-coded score heatmaps.

---

## 🛠️ Tech Stack & Dependencies

* **Framework**: Python 3.11, [FastAPI](https://fastapi.tiangolo.com/) (Async ASGI)
* **Database**: PostgreSQL 15+ with [`pgvector`](https://github.com/pgvector/pgvector) & [SQLAlchemy 2.0 Async](https://www.sqlalchemy.org/)
* **NLP & NER**: [spaCy 3.7](https://spacy.io/) (`en_core_web_sm`)
* **Vector Embeddings**: [SentenceTransformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`, 384-dimensional embeddings)
* **LLM Engine**: [NVIDIA NIM API](https://build.nvidia.com) — Meta Llama 3.3 70B Instruct
* **Document Parser**: `pdfplumber` & `PyMuPDF` (`fitz`)
* **Frontend**: Responsive Single-Page Web Dashboard (Vanilla JS + CSS3 + HTML5)
* **Reporting**: `openpyxl` (Automated Excel report generation)

---

## 🚀 Quick Start Guide

### 1. Prerequisites
* **Python**: `3.11` or higher
* **Database**: PostgreSQL 15+ with `pgvector` extension enabled
* **API Key**: NVIDIA NIM API Key (Free available at [build.nvidia.com](https://build.nvidia.com))

---

### 2. Backend Setup

```bash
# Clone repository
git clone https://github.com/kushwanth-masupalli/smart-resume-screener.git
cd smart-resume-screener/backend

# Create & activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

---

### 3. Environment Configuration

Create a `.env` file inside the `backend/` directory:

```env
# Database Connection (PostgreSQL with pgvector)
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/resume_screener
SYNC_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/resume_screener

# NVIDIA NIM LLM API Config
NVIDIA_API_KEY=nvapi-your-key-here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

# Pipeline Tuning Settings
SIEVE_TOP_K_PERCENT=0.20
SIEVE_MIN_CANDIDATES=5
SIEVE_MAX_CANDIDATES=50

# Models
EMBEDDING_MODEL=all-MiniLM-L6-v2
SPACY_MODEL=en_core_web_sm
LLM_MODEL=meta/llama-3.3-70b-instruct
```

---

### 4. Database Setup

Ensure PostgreSQL is running and create the database:

```sql
CREATE DATABASE resume_screener;
\c resume_screener;
CREATE EXTENSION IF NOT EXISTS vector;
```

---

### 5. Launch Application

Start the FastAPI server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Once running:
* **Web UI Dashboard**: Access directly at [`http://localhost:8000/`](http://localhost:8000/)
* **Interactive OpenAPI Specs**: Explore at [`http://localhost:8000/docs`](http://localhost:8000/docs)

---

## 📡 REST API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/v1/health` | `GET` | System health & status check |
| `/api/v1/resumes/upload` | `POST` | Upload single resume (PDF/TXT) with SHA-256 dedup |
| `/api/v1/resumes/upload-batch` | `POST` | Batch upload multiple candidate resumes |
| `/api/v1/resumes` | `GET` | List all uploaded candidate profiles |
| `/api/v1/resumes/{candidate_id}` | `GET` | Fetch detailed candidate profile & extracted entity JSON |
| `/api/v1/jobs` | `POST` | Create a new Job Description with embedded vectors |
| `/api/v1/jobs` | `GET` | List all created job descriptions |
| `/api/v1/screen/{job_id}` | `POST` | Trigger cascading screening pipeline (`Sieve → Judge → Guard → Coach`) |
| `/api/v1/screen/{job_id}/results` | `GET` | Retrieve candidate screening leaderboard (with filtering & sorting) |
| `/api/v1/screen/{job_id}/export` | `GET` | Download styled Excel report (`.xlsx`) |
| `/api/v1/stats` | `GET` | Retrieve high-level application metrics |

---

## 📂 Project Structure

```
smart-resume-screener/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py           # REST API Route Endpoints
│   │   ├── core/
│   │   │   ├── config.py           # Pydantic Settings & Environment Loader
│   │   │   ├── database.py         # Async SQLAlchemy Engine & Session Management
│   │   │   └── embeddings.py       # Local SentenceTransformers Vector Generator
│   │   ├── models/
│   │   │   ├── database_models.py  # ORM Models (Candidate, JobDescription, Match)
│   │   │   └── schemas.py          # Pydantic Request/Response Schemas
│   │   └── services/
│   │       ├── orchestrator.py     # End-to-End Pipeline Coordinator
│   │       ├── ingest.py           # Document Text Parsing & SHA-256 Hashing
│   │       ├── structuring.py      # spaCy NER & Regex Information Extraction
│   │       ├── sieve.py            # Stage 1: Local Embedding Cosine Pre-Filter
│   │       ├── judge.py            # Stage 2: NVIDIA NIM Llama 3.3 70B JSON Judge
│   │       ├── grounding_guard.py  # Fact-Checking & Hallucination Detector
│   │       ├── blind_mode.py       # PII Anonymization & Redaction Engine
│   │       ├── gap_coach.py        # Interview Probe Question Generator
│   │       └── export.py           # OpenPyXL Styled Excel Generator
│   ├── main.py                     # FastAPI Application Initialization
│   ├── requirements.txt            # Python Dependencies
│   └── README.md                   # Backend Specific Documentation
├── frontend/
│   └── index.html                  # Single-Page Web Dashboard Interface
├── render.yaml                     # Render Cloud Deployment Blueprint
└── README.md                       # Main Repository Documentation
```

---

## 👨‍💻 Candidate Submission Details

This project was built as an **internship technical assessment** to demonstrate production-grade AI system architecture, cost optimization, and full-stack software development skills.

* **Developer**: Masupalli Kushwanth
* **Email**: [masupallikushwanth2005@gmail.com](mailto:masupallikushwanth2005@gmail.com)
* **GitHub Repository**: [smart-resume-screener](https://github.com/kushwanth-masupalli/smart-resume-screener)

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
