"""
Ingest & Parse service — Extract text from PDFs/text files with SHA-256 caching.
"""

import hashlib
import io
from pathlib import Path

import pdfplumber


def compute_file_hash(content: bytes) -> str:
    """Compute SHA-256 hash of file content for deduplication."""
    return hashlib.sha256(content).hexdigest()


def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from a PDF using pdfplumber."""
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_content)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts)


def extract_text_from_txt(file_content: bytes, encoding: str = "utf-8") -> str:
    """Extract text from a plain text file."""
    return file_content.decode(encoding)


def ingest_file(filename: str, file_content: bytes) -> tuple[str, str]:
    """
    Ingest a file and extract its text content.

    Returns:
        (resume_hash, raw_text) tuple
    """
    resume_hash = compute_file_hash(file_content)
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        raw_text = extract_text_from_pdf(file_content)
    elif suffix in (".txt", ".md", ".rtf"):
        raw_text = extract_text_from_txt(file_content)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .txt, .md")

    if not raw_text.strip():
        raise ValueError(f"Could not extract any text from {filename}")

    return resume_hash, raw_text
