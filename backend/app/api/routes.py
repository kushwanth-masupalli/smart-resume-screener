"""
FastAPI routes for the Smart Resume Screener.
"""

from uuid import UUID
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case, or_
from app.core.database import get_db
from app.models.database_models import Candidate, JobDescription, Match
from app.models.schemas import (
    CandidateResponse, JobDescriptionResponse,
    ScreeningParams, ScreeningResult, MatchResponse,
    SieveParams, JudgeScore, GroundingFlag,
)
from app.services.orchestrator import ingest_candidate, ingest_job_description, run_screening

router = APIRouter()


# ── Health ──

@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "Smart Resume Screener"}


# ── Resume Management ──

@router.post("/resumes/upload", response_model=CandidateResponse)
async def upload_resume(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a resume (PDF or text). Parses, structures, embeds, and stores."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        candidate = await ingest_candidate(db, file.filename, content)
        await db.commit()
        return candidate
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/resumes/upload-batch", response_model=list[CandidateResponse])
async def upload_resumes_batch(
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload multiple resumes at once."""
    results = []
    errors = []
    for file in files:
        try:
            content = await file.read()
            if len(content) > 0:
                candidate = await ingest_candidate(db, file.filename, content)
                results.append(candidate)
        except (ValueError, Exception) as e:
            errors.append({"filename": file.filename, "error": str(e)})

    await db.commit()
    return results


@router.get("/resumes", response_model=list[CandidateResponse])
async def list_resumes(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    """List all uploaded resumes."""
    result = await db.execute(
        select(Candidate).offset(skip).limit(limit).order_by(Candidate.created_at.desc())
    )
    return result.scalars().all()


@router.get("/resumes/{candidate_id}")
async def get_resume_detail(candidate_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get full resume details including parsed data."""
    result = await db.execute(select(Candidate).where(Candidate.id == candidate_id))
    candidate = result.scalar_one_or_none()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    return {
        "id": candidate.id,
        "name": candidate.name,
        "email": candidate.email,
        "phone": candidate.phone,
        "resume_hash": candidate.resume_hash,
        "raw_text": candidate.raw_text,
        "parsed_json": candidate.parsed_json,
        "created_at": candidate.created_at,
    }


@router.delete("/resumes/{candidate_id}")
async def delete_resume(candidate_id: UUID, db: AsyncSession = Depends(get_db)):
    """Delete a resume and all associated matches."""
    result = await db.execute(select(Candidate).where(Candidate.id == candidate_id))
    candidate = result.scalar_one_or_none()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    await db.delete(candidate)
    await db.commit()
    return {"deleted": str(candidate_id)}


# ── Job Descriptions ──

@router.post("/jobs", response_model=JobDescriptionResponse)
async def create_job(
    title: str = Form(...),
    text: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """Create a job description."""
    job = await ingest_job_description(db, title, text)
    await db.commit()
    return job


@router.get("/jobs", response_model=list[JobDescriptionResponse])
async def list_jobs(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    """List all job descriptions."""
    result = await db.execute(
        select(JobDescription).offset(skip).limit(limit).order_by(JobDescription.created_at.desc())
    )
    return result.scalars().all()


@router.get("/jobs/{job_id}")
async def get_job_detail(job_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get full job description details."""
    result = await db.execute(select(JobDescription).where(JobDescription.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job description not found")

    return {
        "id": job.id,
        "title": job.title,
        "text": job.text,
        "parsed_json": job.parsed_json,
        "created_at": job.created_at,
    }


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: UUID, db: AsyncSession = Depends(get_db)):
    """Delete a job description and all associated matches."""
    result = await db.execute(select(JobDescription).where(JobDescription.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job description not found")

    await db.delete(job)
    await db.commit()
    return {"deleted": str(job_id)}


# ── Screening ──

@router.post("/screen/{job_id}", response_model=ScreeningResult)
async def screen_candidates(
    job_id: UUID,
    params: ScreeningParams = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Run the full screening pipeline for a job description.
    Sieve → Judge → Grounding Guard → Gap Coach.
    """
    if params is None:
        params = ScreeningParams()

    try:
        result = await run_screening(db, job_id, params)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/screen/{job_id}/results")
async def get_screening_results(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    # Sorting
    sort_by: str = Query("overall_score", description="Sort field: overall_score, skills_match, experience_match, education_match, sieve_score, sieve_rank"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    # Filtering
    min_score: float | None = Query(None, description="Minimum overall score (0-10)", ge=0, le=10),
    max_score: float | None = Query(None, description="Maximum overall score (0-10)", ge=0, le=10),
    min_skills: int | None = Query(None, description="Minimum skills_match (0-10)", ge=0, le=10),
    min_experience: int | None = Query(None, description="Minimum experience_match (0-10)", ge=0, le=10),
    has_red_flags: bool | None = Query(None, description="Filter by red flags presence"),
    has_hallucinations: bool | None = Query(None, description="Filter by hallucination flags presence"),
    blind_mode: bool | None = Query(None, description="Filter by blind mode usage"),
    search: str | None = Query(None, description="Search candidate name (case-insensitive)") ,
    # Pagination
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    """Get existing screening results for a job with sorting, filtering, and pagination."""
    # Allowed sort columns
    sort_columns = {
        "overall_score": Match.overall_score,
        "skills_match": Match.skills_match,
        "experience_match": Match.experience_match,
        "education_match": Match.education_match,
        "sieve_score": Match.sieve_score,
        "sieve_rank": Match.sieve_rank,
    }

    sort_col = sort_columns.get(sort_by, Match.overall_score)
    sort_fn = sort_col.desc() if sort_order.lower() == "desc" else sort_col.asc()

    query = select(Match).where(Match.job_id == job_id)

    # Filters
    if min_score is not None:
        query = query.where(Match.overall_score >= min_score)
    if max_score is not None:
        query = query.where(Match.overall_score <= max_score)
    if min_skills is not None:
        query = query.where(Match.skills_match >= min_skills)
    if min_experience is not None:
        query = query.where(Match.experience_match >= min_experience)
    if blind_mode is not None:
        query = query.where(Match.blind_mode == int(blind_mode))

    # Sort
    query = query.order_by(sort_fn.nullslast())

    # Pagination
    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    matches = result.scalars().all()

    if not matches:
        raise HTTPException(status_code=404, detail="No screening results found for this job")

    job_result = await db.execute(select(JobDescription).where(JobDescription.id == job_id))
    job = job_result.scalar_one_or_none()

    # Post-query filters (for JSON fields not queryable in SQL)
    filtered = []
    for m in matches:
        # Red flag filter
        if has_red_flags is not None:
            judge = m.judge_json or {}
            has_flags = bool(judge.get("red_flags"))
            if has_flags != has_red_flags:
                continue

        # Hallucination filter
        if has_hallucinations is not None:
            flags = m.grounding_flags or []
            has_halluc = any(f.get("confidence") == "high" for f in flags)
            if has_halluc != has_hallucinations:
                continue

        # Name search (requires joining Candidate)
        if search is not None:
            # We'll filter by name in Python since we don't have the join
            # For now, skip this match — we handle it below
            pass

        filtered.append(m)

    # Name search: if needed, load candidate names
    if search and filtered:
        candidate_ids = [m.candidate_id for m in filtered]
        cand_result = await db.execute(
            select(Candidate.id, Candidate.name).where(Candidate.id.in_(candidate_ids))
        )
        name_map = {row[0]: (row[1] or "") for row in cand_result.all()}
        search_lower = search.lower()
        filtered = [m for m in filtered if search_lower in name_map.get(m.candidate_id, "").lower()]

    # ── Tiebreaker Sort ──
    # When two candidates have the same overall_score, use secondary criteria:
    # 1. skills_match (higher = better)
    # 2. experience_match (higher = better)
    # 3. education_match (higher = better)
    # 4. sieve_score (higher = more similar to JD)
    # 5. sieve_rank (lower = better)
    def _sort_key(m):
        return (
            -(m.overall_score or 0),
            -(m.skills_match or 0),
            -(m.experience_match or 0),
            -(m.education_match or 0),
            -(m.sieve_score or 0),
            (m.sieve_rank or 9999),
        )

    filtered.sort(key=_sort_key)

    # Get total count before pagination (approximate)
    total_query = select(func.count(Match.id)).where(Match.job_id == job_id)
    total_count = (await db.execute(total_query)).scalar() or 0

    return {
        "job_id": job_id,
        "job_title": job.title if job else "Unknown",
        "total_matches": len(filtered),
        "total_all_matches": total_count,
        "sort_by": sort_by,
        "sort_order": sort_order,
        "matches": [
            {
                "candidate_id": m.candidate_id,
                "sieve_score": m.sieve_score,
                "sieve_rank": m.sieve_rank,
                "overall_score": m.overall_score,
                "skills_match": m.skills_match,
                "experience_match": m.experience_match,
                "education_match": m.education_match,
                "judge": m.judge_json,
                "grounding_flags": m.grounding_flags,
                "blind_mode": bool(m.blind_mode),
                "processed_at": m.processed_at,
            }
            for m in filtered
        ],
    }


# ── Excel Export ──

from fastapi.responses import StreamingResponse
from app.services.export import generate_screening_excel

@router.get("/screen/{job_id}/export")
async def export_screening_results(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Export all screening results to an Excel file."""
    try:
        excel_buffer = await generate_screening_excel(db, job_id)
        return StreamingResponse(
            excel_buffer,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename=screening_results_{job_id}.xlsx"
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Stats ──

@router.get("/stats")
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Get overall system statistics."""
    candidate_count = (await db.execute(select(func.count(Candidate.id)))).scalar()
    job_count = (await db.execute(select(func.count(JobDescription.id)))).scalar()
    match_count = (await db.execute(select(func.count(Match.id)))).scalar()
    llm_match_count = (await db.execute(
        select(func.count(Match.id)).where(Match.overall_score.isnot(None))
    )).scalar()

    return {
        "total_candidates": candidate_count or 0,
        "total_jobs": job_count or 0,
        "total_matches": match_count or 0,
        "llm_scored_matches": llm_match_count or 0,
    }
