"""
SQLAlchemy database models with pgvector support.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Float, Integer, DateTime, ForeignKey, JSON, LargeBinary
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.core.database import Base
from app.core.config import get_settings

settings = get_settings()


class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True)
    phone = Column(String(100), nullable=True)
    resume_hash = Column(String(64), unique=True, index=True, nullable=False)
    raw_text = Column(Text, nullable=False)
    parsed_json = Column(JSONB, nullable=True)  # Structured extraction results
    embedding = Column(Vector(settings.embedding_dimension), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    matches = relationship("Match", back_populates="candidate", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Candidate {self.name or 'Unknown'} ({self.id})>"


class JobDescription(Base):
    __tablename__ = "job_descriptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    text = Column(Text, nullable=False)
    parsed_json = Column(JSONB, nullable=True)
    embedding = Column(Vector(settings.embedding_dimension), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    matches = relationship("Match", back_populates="job_description", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<JobDescription {self.title} ({self.id})>"


class Match(Base):
    __tablename__ = "matches"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id"), nullable=False, index=True)
    job_id = Column(UUID(as_uuid=True), ForeignKey("job_descriptions.id"), nullable=False, index=True)

    # Stage 1: Sieve
    sieve_score = Column(Float, nullable=True)  # Cosine similarity score
    sieve_rank = Column(Integer, nullable=True)  # Rank within this job's candidates

    # Stage 2: Judge
    judge_json = Column(JSONB, nullable=True)  # Full structured LLM output
    overall_score = Column(Float, nullable=True)  # Extracted for easy sorting
    skills_match = Column(Float, nullable=True)
    experience_match = Column(Float, nullable=True)
    education_match = Column(Float, nullable=True)

    # Grounding Guard
    grounding_flags = Column(JSONB, nullable=True)  # List of flagged claims

    # Metadata
    blind_mode = Column(Integer, default=0)  # 0 = normal, 1 = blind mode
    processed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    candidate = relationship("Candidate", back_populates="matches")
    job_description = relationship("JobDescription", back_populates="matches")

    def __repr__(self):
        return f"<Match candidate={self.candidate_id} job={self.job_id} score={self.overall_score}>"
