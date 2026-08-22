"""
Orchestrator — Ties the full screening pipeline together.
Phases: Ingest → Structure → Embed → Sieve → Judge → Ground
"""

import asyncio
from datetime import datetime, timezone
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models.database_models import Candidate, JobDescription, Match
from app.models.schemas import (
    MatchResponse, ScreeningResult, ScreeningParams, JudgeScore, GroundingFlag,
)
from app.services.ingest import ingest_file, compute_file_hash
from app.services.structuring import structure_resume
from app.services.sieve import sieve, SieveResult
from app.services.judge import call_judge
from app.services.grounding_guard import ground_check, get_hallucination_summary
from app.services.blind_mode import blindify
from app.core.embeddings import embed_text


async def ingest_candidate(
    db: AsyncSession,
    filename: str,
    file_content: bytes,
) -> Candidate:
    """
    Ingest a resume file: parse, structure, embed, and store.
    Returns existing candidate if resume_hash already exists (cache hit).
    """
    raw_hash, raw_text = ingest_file(filename, file_content)

    # Check cache — if this exact resume was already parsed, reuse it
    result = await db.execute(
        select(Candidate).where(Candidate.resume_hash == raw_hash)
    )
    existing = result.scalar_one_or_none()
    if existing:
        return existing

    # Structure the resume (spaCy + regex, no LLM)
    parsed_data = structure_resume(raw_text)

    # Embed the structured summary
    summary = parsed_data.get("structured_summary", raw_text[:500])
    embedding = embed_text(summary)

    # Create candidate record
    candidate = Candidate(
        name=parsed_data.get("name"),
        email=parsed_data.get("email"),
        phone=parsed_data.get("phone"),
        resume_hash=raw_hash,
        raw_text=raw_text,
        parsed_json=parsed_data,
        embedding=embedding,
    )
    db.add(candidate)
    await db.flush()
    return candidate


async def ingest_job_description(
    db: AsyncSession,
    title: str,
    raw_text: str,
) -> JobDescription:
    """Ingest a job description: embed and store."""
    embedding = embed_text(raw_text)

    job = JobDescription(
        title=title,
        raw_text=raw_text,
        embedding=embedding,
    )
    db.add(job)
    await db.flush()
    return job


async def _process_candidate_match(
    candidate: Candidate,
    rank: int,
    sieve_score: float,
    is_survivor: bool,
    job: JobDescription,
    params: ScreeningParams,
) -> tuple[Match, MatchResponse]:
    """Process a single candidate: call Judge asynchronously if survivor."""
    match = Match(
        candidate_id=candidate.id,
        job_id=job.id,
        sieve_score=round(sieve_score, 4),
        sieve_rank=rank,
        blind_mode=1 if params.blind_mode else 0,
    )

    match_resp = MatchResponse(
        candidate_id=candidate.id,
        candidate_name=candidate.name,
        candidate_email=candidate.email,
        sieve_score=round(sieve_score, 4),
        sieve_rank=rank,
        blind_mode=params.blind_mode,
    )

    if is_survivor:
        judge_text = candidate.raw_text
        parsed_data = candidate.parsed_json or {}

        if params.blind_mode:
            judge_text, _ = blindify(candidate.raw_text, parsed_data)

        # Call Judge asynchronously
        judge_score = await call_judge(
            jd_text=job.raw_text,
            resume_text=judge_text,
            resume_name="[REDACTED]" if params.blind_mode else (candidate.name or "Candidate"),
        )
        match_resp.judge = judge_score

        match.judge_json = judge_score.model_dump()
        match.overall_score = judge_score.overall_score
        match.skills_match = judge_score.skills_match
        match.experience_match = judge_score.experience_match
        match.education_match = judge_score.education_match

        # Grounding Guard
        extracted_skills = parsed_data.get("skills", [])
        flags = ground_check(judge_score, extracted_skills)
        match.grounding_flags = [f.model_dump() for f in flags]
        match_resp.grounding_flags = flags

    match.processed_at = datetime.utcnow()
    return match, match_resp


async def run_screening(
    db: AsyncSession,
    job_id: UUID,
    params: ScreeningParams,
) -> ScreeningResult:
    """
    Run the full screening pipeline for a job against all candidates concurrently.
    """
    start_time =  datetime.now(timezone.utc)

    # Load job description
    result = await db.execute(select(JobDescription).where(JobDescription.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise ValueError(f"Job description {job_id} not found")

    # Load all candidates
    result = await db.execute(select(Candidate).options(selectinload(Candidate.matches)))
    candidates = list(result.scalars().all())

    if not candidates:
        return ScreeningResult(
            job_id=job.id,
            job_title=job.title,
            total_candidates=0,
            sieve_survivors=0,
            llm_calls_made=0,
            cost_reduction_percent=100.0,
            matches=[],
            screening_time_seconds=round((datetime.now(timezone.utc) - start_time).total_seconds(), 2),
            created_at=job.created_at,
        )

    # ── Stage 1: The Sieve ──
    summaries = []
    for c in candidates:
        parsed = c.parsed_json or {}
        summary = parsed.get("structured_summary", c.raw_text[:500])
        summaries.append(summary)

    sieve_result: SieveResult = sieve(
        jd_text=job.raw_text,
        candidate_summaries=summaries,
        top_k_percent=params.sieve_params.top_k_percent,
        min_candidates=params.sieve_params.min_candidates,
        max_candidates=params.sieve_params.max_candidates,
    )

    # ── Stage 2: The Judge + Grounding Guard (Concurrent Execution) ──
    tasks = []
    for rank, (candidate_idx, sieve_score) in enumerate(sieve_result.ranked_candidates, 1):
        candidate = candidates[candidate_idx]
        is_survivor = candidate_idx in sieve_result.top_k_indices
        tasks.append(
            _process_candidate_match(
                candidate=candidate,
                rank=rank,
                sieve_score=sieve_score,
                is_survivor=is_survivor,
                job=job,
                params=params,
            )
        )

    processed_pairs = await asyncio.gather(*tasks)

    match_responses: list[MatchResponse] = []
    for match, match_resp in processed_pairs:
        db.add(match)
        match_responses.append(match_resp)

    await db.commit()

    # ── Tiebreaker Sort ──
    # When two candidates have the same overall_score, use secondary criteria:
    # 1. skills_match (higher = better)
    # 2. experience_match (higher = better)
    # 3. education_match (higher = better)
    # 4. sieve_score (higher = more similar to JD)
    # 5. sieve_rank (lower = better)
    def _sort_key(m: MatchResponse):
        judge = m.judge
        return (
            -(judge.overall_score or 0),
            -(judge.skills_match or 0),
            -(judge.experience_match or 0),
            -(judge.education_match or 0),
            -(m.sieve_score or 0),
            (m.sieve_rank or 9999),
        )

    match_responses.sort(key=_sort_key)

    elapsed = round((datetime.now(timezone.utc) - start_time).total_seconds(), 2)

    return ScreeningResult(
        job_id=job.id,
        job_title=job.title,
        total_candidates=sieve_result.total_candidates,
        sieve_survivors=sieve_result.survivors_count,
        llm_calls_made=sieve_result.survivors_count,
        cost_reduction_percent=sieve_result.cost_reduction_percent,
        matches=match_responses,
        screening_time_seconds=elapsed,
        created_at=job.created_at,
    )
