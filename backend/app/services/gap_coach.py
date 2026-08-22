"""
Gap Coach — Generates targeted interview questions for shortlisted candidates.

For each candidate, analyzes their weak areas (missing skills, low sub-scores)
and generates 3 interview questions mixing behavioral and technical.
"""

import json
import openai
from app.core.config import get_settings

settings = get_settings()


async def generate_questions(
    jd_text: str,
    resume_text: str,
    matched_skills: list[str],
    missing_skills: list[str],
    low_scoring_dims: list[str],
    num_questions: int = 3,
) -> list[dict]:
    """
    Generate targeted interview questions for a candidate.

    Returns a list of dicts with keys: question, focus_area, type
    """
    # Build the prompt
    focus_areas = []
    if missing_skills:
        focus_areas.append(f"Missing skills: {', '.join(missing_skills)}")
    if low_scoring_dims:
        focus_areas.append(f"Weak areas: {', '.join(low_scoring_dims)}")

    focus_text = "; ".join(focus_areas) if focus_areas else "general fit assessment"

    system_prompt = (
        "You are an expert technical interviewer. Generate concise, targeted "
        "interview questions based on a candidate's profile and the gaps identified "
        "between their skills and the job requirements.\n\n"
        "Return ONLY a JSON array of objects, each with:\n"
        '- "question": the interview question (string)\n'
        '- "focus_area": what this question targets (string)\n'
        '- "type": "technical" or "behavioral" (string)\n\n'
        f"Generate exactly {num_questions} questions."
    )

    user_prompt = (
        f"Job Description:\n{jd_text[:2000]}\n\n"
        f"Candidate Resume (excerpt):\n{resume_text[:1500]}\n\n"
        f"Matched Skills: {', '.join(matched_skills) if matched_skills else 'None'}\n"
        f"Focus Areas: {focus_text}\n\n"
        f"Generate {num_questions} targeted interview questions."
    )

    try:
        client = openai.AsyncOpenAI(
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
        )

        response = await client.chat.completions.create(
            model=settings.llm_model,
            temperature=0.4,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = response.choices[0].message.content.strip()
        # Try to extract JSON from the response
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        questions = json.loads(content)
        if isinstance(questions, list):
            return questions[:num_questions]

    except Exception as e:
        print(f"Gap Coach error: {e}")

    # Fallback: generate generic questions
    return [
        {
            "question": "Can you walk me through a challenging project you've worked on and your role in it?",
            "focus_area": "general experience",
            "type": "behavioral",
        },
        {
            "question": "How do you approach learning new technologies or skills that are outside your comfort zone?",
            "focus_area": "learning ability",
            "type": "behavioral",
        },
        {
            "question": "Describe a situation where you had to collaborate with a team to solve a complex technical problem.",
            "focus_area": "teamwork and problem solving",
            "type": "behavioral",
        },
    ][:num_questions]
