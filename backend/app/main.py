"""
Smart Resume Screener — Main FastAPI Application.

Architecture:
  Ingest → Structure → Sieve (embeddings) → Judge (LLM) → Grounding Guard → Gap Coach
  Only the Judge stage uses LLM calls, and only on Sieve survivors.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import init_db, close_db
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: create tables
    await init_db()
    yield
    # Shutdown: close connections
    await close_db()


app = FastAPI(
    title="Smart Resume Screener",
    description="""
    An efficient resume screening system that uses local embeddings for pre-filtering
    and LLM calls only on the most promising candidates.

    Key features:
    - **The Sieve**: Local embeddings pre-filter candidates (zero LLM cost)
    - **The Judge**: Structured LLM scoring with forced JSON schema
    - **Grounding Guard**: Fact-checks LLM claims against extracted data
    - **Blind Mode**: Strips identity signals before scoring for bias mitigation
    - **Gap Coach**: Generates targeted interview questions for shortlisted candidates
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow local dev, file:// origin, and any Render frontend URL
import os
_cors_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:8080",
    "null",  # file:// origin (opening index.html directly)
]
# Add Render frontend URL if set
if frontend_url := os.environ.get("FRONTEND_URL"):
    _cors_origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from pathlib import Path
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

@app.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")
# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "name": "Smart Resume Screener",
        "version": "1.0.0",
        "docs": "/docs",
        "architecture": {
            "stage_1_sieve": "Local embeddings pre-filter (zero LLM cost)",
            "stage_2_judge": "Structured LLM scoring (only on survivors)",
            "grounding_guard": "Fact-checks LLM claims",
            "blind_mode": "Bias mitigation through identity redaction",
            "gap_coach": "Interview question generation",
        },
    }
