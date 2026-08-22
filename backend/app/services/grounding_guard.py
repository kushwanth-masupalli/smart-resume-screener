"""
Grounding Guard — Cross-check the LLM's claims against extracted facts.
Catches hallucinations where the Judge claims a skill that isn't actually in the resume.
"""

from difflib import SequenceMatcher
from app.models.schemas import JudgeScore, GroundingFlag


def fuzzy_match(claimed: str, extracted: list[str], threshold: float = 0.6) -> tuple[bool, str]:
    """
    Check if a claimed skill is supported by extracted skills using fuzzy matching.
    Returns (is_supported, best_match).
    """
    claimed_lower = claimed.lower().strip()

    # Exact match
    for skill in extracted:
        if claimed_lower == skill.lower().strip():
            return True, skill

    # Substring match
    for skill in extracted:
        skill_lower = skill.lower().strip()
        if claimed_lower in skill_lower or skill_lower in claimed_lower:
            return True, skill

    # Fuzzy match
    best_ratio = 0.0
    best_match = ""
    for skill in extracted:
        ratio = SequenceMatcher(None, claimed_lower, skill.lower().strip()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = skill

    if best_ratio >= threshold:
        return True, best_match

    return False, best_match


def ground_check(
    judge_output: JudgeScore,
    extracted_skills: list[str],
) -> list[GroundingFlag]:
    """
    Verify each matched skill from the Judge against the extracted skill list.
    Returns a list of flags for skills that couldn't be verified.
    """
    flags = []

    for skill_claimed in judge_output.matched_skills:
        is_supported, best_match = fuzzy_match(skill_claimed, extracted_skills)

        if not is_supported:
            # The LLM claimed a skill that isn't in the extracted data
            flags.append(GroundingFlag(
                skill_claimed=skill_claimed,
                confidence="high",
                reason=f"Skill '{skill_claimed}' was claimed by the LLM but not found in the structured extraction of the resume.",
            ))
        elif best_match.lower().strip() != skill_claimed.lower().strip():
            # Partial/fuzzy match — lower confidence
            flags.append(GroundingFlag(
                skill_claimed=skill_claimed,
                confidence="medium",
                reason=f"Claimed skill '{skill_claimed}' partially matches extracted skill '{best_match}' (fuzzy match).",
            ))

    return flags


def get_hallucination_summary(flags: list[GroundingFlag]) -> dict:
    """Summarize grounding flags into a digestible format."""
    high_confidence = [f for f in flags if f.confidence == "high"]
    medium_confidence = [f for f in flags if f.confidence == "medium"]

    return {
        "total_flags": len(flags),
        "likely_hallucinations": len(high_confidence),
        "uncertain_matches": len(medium_confidence),
        "hallucinated_skills": [f.skill_claimed for f in high_confidence],
        "uncertain_skills": [f.skill_claimed for f in medium_confidence],
        "trust_score": max(0, 1.0 - (len(high_confidence) * 0.2) - (len(medium_confidence) * 0.05)),
    }
