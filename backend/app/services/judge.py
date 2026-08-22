"""
The Judge (Stage 2) — LLM scoring with forced structured JSON output.
Only called on Sieve survivors. Every call returns chartable, auditable data.
"""

import asyncio
import json
import logging
from openai import AsyncOpenAI
from app.core.config import get_settings
from app.models.schemas import JudgeScore

logger = logging.getLogger(__name__)
settings = get_settings()

client = AsyncOpenAI(
    base_url=settings.nvidia_base_url,
    api_key=settings.nvidia_api_key,
)

JUDGE_SYSTEM_PROMPT = """You are an expert technical recruiter scoring a candidate against a job description.
You must respond ONLY with valid JSON matching this exact schema:
{
  "skills_match": <int 0-10>,
  "experience_match": <int 0-10>,
  "education_match": <int 0-10>,
  "overall_score": <int 0-10>,
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "justification": "2-3 specific sentences explaining the score",
  "red_flags": ["flag1"] or []
}

Rules:
- Base EVERY claim strictly on the resume text provided.
- Do NOT infer skills that are not explicitly stated or clearly implied by listed experience.
- Be specific in your justification — reference actual companies, projects, or technologies.
- List concrete missing skills that the JD requires but the resume doesn't mention.
- Red flags: gaps in employment, mismatched seniority, unverifiable claims.
- Be calibrated: most candidates should score 3-7, not 9-10.
"""


async def call_judge(jd_text: str, resume_text: str, resume_name: str = "Candidate", max_retries: int = 3) -> JudgeScore:
    """
    Call The Judge LLM asynchronously to score a candidate against a job description.
    Includes retry logic with backoff. Returns structured JudgeScore.
    """
    user_message = f"""Job Description:
---
{jd_text}
---

Candidate Resume ({resume_name}):
---
{resume_text[:8000]}
---

Score this candidate against the job description. Respond ONLY with valid JSON."""

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=1024,
            )

            # Safely extract text from response
            response_text = response.choices[0].message.content.strip() if response.choices else ""
            if not response_text:
                raise ValueError("NVIDIA API returned an empty response.")

            # Extract JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            try:
                judge_data = json.loads(response_text)
            except json.JSONDecodeError:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    judge_data = json.loads(response_text[start:end])
                else:
                    raise ValueError(f"Could not parse Judge response as JSON: {response_text[:200]}")

            return JudgeScore(
                skills_match=int(judge_data.get("skills_match", 0)),
                experience_match=int(judge_data.get("experience_match", 0)),
                education_match=int(judge_data.get("education_match", 0)),
                overall_score=int(judge_data.get("overall_score", 0)),
                matched_skills=judge_data.get("matched_skills", []),
                missing_skills=judge_data.get("missing_skills", []),
                justification=judge_data.get("justification", ""),
                red_flags=judge_data.get("red_flags", []),
            )

        except Exception as e:
            last_error = e
            logger.warning(f"Judge LLM call attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                await asyncio.sleep(1.0 * (2 ** (attempt - 1)))

    raise ValueError(f"Judge LLM scoring failed after {max_retries} retries: {last_error}")

