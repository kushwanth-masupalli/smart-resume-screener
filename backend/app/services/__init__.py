"""
Services package — the core pipeline components.
"""

from app.services.ingest import ingest_file, compute_file_hash
from app.services.structuring import structure_resume
from app.services.sieve import sieve, SieveResult
from app.services.judge import call_judge
from app.services.grounding_guard import ground_check, get_hallucination_summary
from app.services.blind_mode import blindify, deblindify
from app.services.gap_coach import generate_questions
from app.services.orchestrator import (
    ingest_candidate,
    ingest_job_description,
    run_screening,
)

__all__ = [
    "ingest_file",
    "compute_file_hash",
    "structure_resume",
    "sieve",
    "SieveResult",
    "call_judge",
    "ground_check",
    "get_hallucination_summary",
    "blindify",
    "deblindify",
    "generate_questions",
    "ingest_candidate",
    "ingest_job_description",
    "run_screening",
]
