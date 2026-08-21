"""
Structuring Layer — Extract structured data from resume text using spaCy + regex.
No LLM calls. Deterministic, fast, free.
"""

from __future__ import annotations

import re
from typing import Any

import spacy
from app.core.config import get_settings

settings = get_settings()

_nlp = None


def get_nlp() -> spacy.language.Language:
    """Lazy-load spaCy model with auto-download fallback."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(settings.spacy_model)
        except OSError:
            from spacy.cli import download
            download(settings.spacy_model)
            _nlp = spacy.load(settings.spacy_model)
    return _nlp


# ── Common skills taxonomy ──

COMMON_SKILLS = {
    # Programming Languages
    "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust", "ruby",
    "php", "swift", "kotlin", "scala", "r", "matlab", "sql", "html", "css",
    # Frameworks & Libraries
    "react", "vue", "angular", "node.js", "nodejs", "django", "flask", "fastapi",
    "express", "spring", "spring boot", "rails", "laravel", "next.js", "nextjs",
    "svelte", "tailwind", "bootstrap", "jquery", "pytorch", "tensorflow",
    "keras", "scikit-learn", "pandas", "numpy", "scipy",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "k8s", "terraform", "ansible",
    "jenkins", "github actions", "ci/cd", "nginx", "linux", "bash", "git",
    # Databases
    "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
    "sqlite", "dynamodb", "cassandra", "neo4j", "sqlalchemy", "prisma",
    # Data & ML
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "data science", "data analysis", "data engineering",
    "spark", "hadoop", "kafka", "airflow", "dbt",
    # Design & Product
    "figma", "sketch", "adobe xd", "photoshop", "illustrator",
    # Other
    "agile", "scrum", "jira", "confluence", "rest api", "graphql", "grpc",
    "microservices", "serverless", "lambda", "api design",
}

# Aliases → canonical name
SKILL_ALIASES = {
    "js": "javascript", "javascript": "javascript",
    "ts": "typescript", "typescript": "typescript",
    "py": "python", "python3": "python",
    "node": "node.js", "nodejs": "node.js",
    "k8s": "kubernetes", "k8": "kubernetes",
    "postgres": "postgresql", "pg": "postgresql",
    "tf": "tensorflow", "torch": "pytorch",
    "sklearn": "scikit-learn", "scikit learn": "scikit-learn",
    "ci cd": "ci/cd", "ci/cd": "ci/cd",
    "rest": "rest api", "rest api": "rest api",
    "dl": "deep learning", "ml": "machine learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
}


def normalize_skill(skill: str) -> str:
    """Normalize a skill string to its canonical form."""
    lower = skill.strip().lower()
    return SKILL_ALIASES.get(lower, lower)


def extract_contact_info(doc: spacy.tokens.Doc) -> dict[str, str | None]:
    """Extract email, phone from text using regex."""
    text = doc.text

    # Email
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    email = email_match.group(0) if email_match else None

    # Phone (US format patterns)
    phone_match = re.search(
        r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text
    )
    phone = phone_match.group(0).strip() if phone_match else None

    return {"email": email, "phone": phone}


def extract_name(doc: spacy.tokens.Doc) -> str | None:
    """Extract name using spaCy NER — look for PERSON entities."""
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Take the first person entity (likely the resume owner)
            return ent.text.strip()
    return None


def extract_education(doc: spacy.tokens.Doc) -> list[dict[str, Any]]:
    """Extract education entries using patterns."""
    education = []
    text = doc.text
    lines = text.split("\n")

    # Common education keywords
    edu_keywords = {
        "bachelor", "b.s.", "b.a.", "btech", "b.tech",
        "master", "m.s.", "m.a.", "mtech", "m.tech",
        "phd", "ph.d", "doctorate", "mba",
        "associate", "diploma", "certificate",
        "university", "college", "institute", "school",
    }

    # Degree field mapping
    degree_patterns = [
        (r'\b(?:B\.?S\.?|Bachelor\w* of Science)\b', "Bachelor of Science"),
        (r'\b(?:B\.?A\.?|Bachelor\w* of Arts)\b', "Bachelor of Arts"),
        (r'\b(?:M\.?S\.?|Master\w* of Science)\b', "Master of Science"),
        (r'\b(?:M\.?A\.?|Master\w* of Arts)\b', "Master of Arts"),
        (r'\b(?:Ph\.?D\.?|Doctor\w*)\b', "PhD"),
        (r'\bMBA\b', "MBA"),
        (r'\b(?:B\.?Tech|Bachelor\w* of Technology)\b', "B.Tech"),
        (r'\b(?:M\.?Tech|Master\w* of Technology)\b', "M.Tech"),
    ]

    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in edu_keywords):
            # Try to extract a degree
            degree = None
            for pattern, name in degree_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    degree = name
                    break

            # Try to extract field of study
            field = None
            field_match = re.search(
                r'(?:in|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:and|&)\s+[A-Z][a-z]+)*)',
                line
            )
            if field_match:
                field = field_match.group(1)

            education.append({
                "degree": degree,
                "field": field,
                "text": line.strip(),
            })

    return education


def extract_skills(doc: spacy.tokens.Doc) -> list[str]:
    """Extract skills from resume text using keyword matching and NER."""
    text_lower = doc.text.lower()
    found_skills = set()

    # Direct keyword matching against known skills taxonomy
    for skill in COMMON_SKILLS:
        # Use word boundary matching to avoid false positives
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(normalize_skill(skill))

    # Also check for abbreviations in parentheses: e.g., "React (JS)"
    paren_pattern = re.findall(r'\(([A-Za-z0-9/+.\s-]+)\)', doc.text)
    for match in paren_pattern:
        for word in match.split():
            word_lower = word.strip().lower()
            if word_lower in SKILL_ALIASES or word_lower in COMMON_SKILLS:
                found_skills.add(normalize_skill(word_lower))

    return sorted(found_skills)


def extract_dates(doc: spacy.tokens.Doc) -> list[dict[str, str]]:
    """Extract date ranges (e.g., employment/education periods)."""
    dates = []
    text = doc.text

    # Patterns for date ranges
    date_range_patterns = [
        # "Jan 2020 - Present" or "January 2020 - Present"
        r'([A-Z][a-z]+\.?\s+\d{4})\s*[-–—to]+\s*(Present|Current|Now|[A-Z][a-z]+\.?\s+\d{4})',
        # "2020 - 2023" or "2020–2023"
        r'(\d{4})\s*[-–—to]+\s*(Present|Current|Now|\d{4})',
        # "01/2020 - 12/2023"
        r'(\d{1,2}/\d{4})\s*[-–—to]+\s*(Present|Current|Now|\d{1,2}/\d{4})',
    ]

    for pattern in date_range_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            dates.append({
                "start": match.group(1),
                "end": match.group(2),
                "text": match.group(0),
            })

    return dates


def structure_resume(raw_text: str) -> dict[str, Any]:
    """
    Main structuring function. Extracts structured data from resume text.
    Uses spaCy NER + regex. No LLM calls.
    """
    nlp = get_nlp()
    doc = nlp(raw_text)

    contact = extract_contact_info(doc)
    name = extract_name(doc)
    skills = extract_skills(doc)
    education = extract_education(doc)
    dates = extract_dates(doc)

    # Build structured summary for embedding (concise text, not full resume)
    summary_parts = []
    if name:
        summary_parts.append(f"Name: {name}")
    if skills:
        summary_parts.append(f"Skills: {', '.join(skills)}")
    if education:
        edu_strs = []
        for e in education:
            parts = [e.get("degree", ""), e.get("field", "")]
            edu_strs.append(" ".join(p for p in parts if p))
        summary_parts.append(f"Education: {'; '.join(edu_strs)}")
    if dates:
        summary_parts.append(f"Experience periods: {len(dates)} listed")

    structured_summary = " | ".join(summary_parts) if summary_parts else raw_text[:500]

    return {
        "name": name,
        "email": contact.get("email"),
        "phone": contact.get("phone"),
        "skills": skills,
        "education": education,
        "date_ranges": dates,
        "structured_summary": structured_summary,
    }
