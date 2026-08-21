"""
Embedding service using sentence-transformers (runs locally, zero API cost).
"""

import os
os.environ["USE_TORCH"] = "1"  # Force PyTorch backend, skip TensorFlow
os.environ["TRANSFORMERS_NO_TF"] = "1"

import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from app.core.config import get_settings

settings = get_settings()

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_text(text: str) -> list[float]:
    """Embed a single text string."""
    model = get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple text strings in batch (more efficient)."""
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return embeddings.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-8))


def rank_by_similarity(query_embedding: list[float], candidate_embeddings: list[list[float]]) -> list[tuple[int, float]]:
    """
    Rank candidates by cosine similarity to query.
    Returns list of (index, score) tuples sorted by descending score.
    """
    query_arr = np.array(query_embedding, dtype=np.float32)
    candidate_arr = np.array(candidate_embeddings, dtype=np.float32)

    # Normalize for cosine similarity (embeddings are already normalized, but be safe)
    query_norm = query_arr / (np.linalg.norm(query_arr) + 1e-8)
    candidate_norms = candidate_arr / (np.linalg.norm(candidate_arr, axis=1, keepdims=True) + 1e-8)

    scores = np.dot(candidate_norms, query_norm)
    ranked = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)
    return ranked
