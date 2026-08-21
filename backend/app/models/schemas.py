"""
Pydantic schemas for API requests and responses.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID
from typing import Optional


# ── Request Schemas ──

class JobDescriptionUpload(BaseModel):
    title: str = Field(..., description="Job title")
    text: str = Field(..., description="Full job description text")


class SieveParams(BaseModel):
    top_k_percent: float = Field(0.20, ge=0.01, le=1.0, description="Percentage of candidates to pass to Judge")
    min_candidates: int = Field(5, ge=1, description="Minimum candidates to send to Judge")
    max_candidates: int = Field(50, ge=1, description="Maximum candidates to send to Judge")


class ScreeningParams(BaseModel):
    sieve_params: SieveParams = Field(default_factory=SieveParams)
    blind_mode: bool = Field(False, description="Strip identity signals before LLM scoring")


# ── Response Schemas ──

class CandidateResponse(BaseModel):
    id: UUID
    name: Optional[str] = None
    email: Optional[str] = None
    resume_hash: str
    created_at: datetime

    model_config = {"from_attributes": True}


class JobDescriptionResponse(BaseModel):
    id: UUID
    title: str
    text: str
    created_at: datetime
    model_config = {"from_attributes": True, "populate_by_name": True}

class JudgeScore(BaseModel):
    """Structured output from The Judge (LLM scoring)."""
    skills_match: int = Field(..., ge=0, le=10)
    experience_match: int = Field(..., ge=0, le=10)
    education_match: int = Field(..., ge=0, le=10)
    overall_score: int = Field(..., ge=0, le=10)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    justification: str = Field(..., description="2-3 sentence justification")
    red_flags: list[str] = Field(default_factory=list)


class GroundingFlag(BaseModel):
    """A claim from the LLM that couldn't be verified against extracted data."""
    skill_claimed: str
    confidence: str = Field(..., description="high, medium, or low")
    reason: str


class MatchResponse(BaseModel):
    """Full match result for a candidate against a job."""
    candidate_id: UUID
    candidate_name: Optional[str] = None
    candidate_email: Optional[str] = None

    # Stage 1: Sieve
    sieve_score: Optional[float] = None
    sieve_rank: Optional[int] = None

    # Stage 2: Judge
    judge: Optional[JudgeScore] = None

    # Grounding Guard
    grounding_flags: list[GroundingFlag] = Field(default_factory=list)

    # Metadata
    blind_mode: bool = False
    processed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ScreeningResult(BaseModel):
    """Complete screening result for a job posting."""
    job_id: UUID
    job_title: str
    total_candidates: int
    sieve_survivors: int
    llm_calls_made: int
    cost_reduction_percent: float = Field(..., description="Percentage of LLM calls saved vs brute force")
    matches: list[MatchResponse]
    screening_time_seconds: float
    created_at: datetime


class ScreeningStatus(BaseModel):
    """Real-time status update during screening."""
    status: str = Field(..., description="parsing, structuring, sieving, judging, grounding, complete, error")
    progress_percent: int = Field(0, ge=0, le=100)
    message: str = ""
    current_step: str = ""
    total_candidates: int = 0
    sieved_candidates: int = 0
