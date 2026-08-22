"""
Blind Mode — Strip identity signals before LLM scoring to mitigate bias.
Reveals identity only after scoring, for display.
"""

import re


# Patterns for identity signals to redact
IDENTITY_PATTERNS = [
    # Name patterns (spaCy-identified names are handled at the API level,
    # here we handle regex patterns)
    (r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[REDACTED_NAME]'),

    # Email
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]'),

    # Phone
    (r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[REDACTED_PHONE]'),

    # Address patterns (US-style street addresses)
    (r'\d{1,5}\s+[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct)\.?\s*,?\s*[\w\s]*,?\s*[A-Z]{2}\s+\d{5}', '[REDACTED_ADDRESS]'),

    # Graduation years (age proxy) — remove years near education keywords
    (r'((?:graduated?|degree|diploma|university|college|school|bachelor|master|phd|b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?)\s+(?:in\s+\d{4}\s+)?(?:from\s+)?[\w\s]+?)\b(19\d{2}|20[0-2]\d)\b', r'\1[REDACTED_YEAR]'),

    # Standalone graduation years
    (r'\b(19\d{2}|20[0-2]\d)\b', '[REDACTED_YEAR]'),

    # Pronouns (he/him, she/her, they/them)
    (r'\b(?:he/him|she/her|they/them|he\s*/\s*him|she\s*/\s*her|they\s*/\s*them)\b', '[REDACTED_PRONOUN]'),

    # Gender-specific titles
    (r'\b(?:Mr\.|Mrs\.|Ms\.|Miss)\b', '[REDACTED_TITLE]'),

    # LinkedIn / GitHub / personal website (can reveal identity)
    (r'https?://(?:www\.)?linkedin\.com/in/[\w-]+', '[REDACTED_LINKEDIN]'),
    (r'https?://(?:www\.)?github\.com/[\w-]+', '[REDACTED_GITHUB]'),
    (r'https?://[\w.-]+\.\w+(?:/[\w-]*)*', '[REDACTED_URL]'),

    # Marital status / nationality indicators
    (r'\b(?:married|single|divorced|widowed|nationality|citizenship)\b', '[REDACTED_PERSONAL]'),
]


def blindify(raw_text: str, parsed_data: dict | None = None) -> tuple[str, dict]:
    """
    Strip identity signals from resume text for bias-free scoring.

    Args:
        raw_text: The original resume text
        parsed_data: Optional structured data to redact names from

    Returns:
        (redacted_text, redaction_log) tuple
    """
    redacted = raw_text
    log = {"redactions": []}

    # Apply all patterns
    for pattern, replacement in IDENTITY_PATTERNS:
        matches = re.finditer(pattern, redacted, re.IGNORECASE)
        for match in matches:
            log["redactions"].append({
                "type": "regex",
                "original": match.group(0)[:50],  # Truncate for log
                "pattern": pattern[:50],
            })
        redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)

    # Redact name from parsed data if provided
    if parsed_data and parsed_data.get("name"):
        name = parsed_data["name"]
        name_escaped = re.escape(name)
        name_matches = re.findall(name_escaped, redacted)
        for _ in name_matches:
            log["redactions"].append({
                "type": "name",
                "original": name,
                "pattern": "spaCy NER name",
            })
        redacted = re.sub(name_escaped, '[REDACTED_NAME]', redacted)

    # Collapse multiple blank lines left by redaction
    redacted = re.sub(r'\n{3,}', '\n\n', redacted)

    return redacted, log


def deblindify(scored_text: str, original_text: str, redaction_log: dict) -> str:
    """
    After scoring, replace redacted placeholders with original values.
    """
    result = scored_text
    for redaction in redaction_log.get("redactions", []):
        # This is a simplified approach; in production you'd store offsets
        pass

    # For display purposes, we typically just return the original text
    # with the scores attached, not the redacted version
    return original_text
