"""
The Sieve (Stage 1) — Local embedding pre-filter. ZERO LLM calls.
Embed the JD once, rank all candidates by cosine similarity, take top-K.
"""

from dataclasses import dataclass
from app.core.embeddings import embed_text, embed_texts, rank_by_similarity
from app.core.config import get_settings

settings = get_settings()


@dataclass
class SieveResult:
    """Result from the Sieve stage."""
    ranked_candidates: list[tuple[int, float]]  # (index_in_batch, score)
    top_k_indices: list[int]  # Indices of survivors passed to Judge
    total_candidates: int
    survivors_count: int
    cost_reduction_percent: float  # % of LLM calls saved


def compute_top_k(total: int, percent: float, min_k: int, max_k: int) -> int:
    """Compute how many candidates to pass to the Judge."""
    k = max(min_k, int(total * percent))
    return min(k, max_k, total)


def sieve(
    jd_text: str,
    candidate_summaries: list[str],
    top_k_percent: float | None = None,
    min_candidates: int | None = None,
    max_candidates: int | None = None,
) -> SieveResult:
    """
    Run The Sieve: embed JD, embed all candidates, rank by cosine similarity.

    This is the efficiency core — it narrows N candidates to K survivors
    with zero LLM calls, saving 80-90% of LLM cost.
    """
    percent = top_k_percent or settings.sieve_top_k_percent
    min_k = min_candidates or settings.sieve_min_candidates
    max_k = max_candidates or settings.sieve_max_candidates

    if not candidate_summaries:
        return SieveResult(
            ranked_candidates=[],
            top_k_indices=[],
            total_candidates=0,
            survivors_count=0,
            cost_reduction_percent=100.0,
        )

    # Step 1: Embed the JD (one call)
    jd_embedding = embed_text(jd_text)

    # Step 2: Embed all candidate summaries (batched, local, fast)
    candidate_embeddings = embed_texts(candidate_summaries)

    # Step 3: Rank by cosine similarity
    ranked = rank_by_similarity(jd_embedding, candidate_embeddings)

    # Step 4: Select top-K
    k = compute_top_k(len(candidate_summaries), percent, min_k, max_k)
    top_k = [idx for idx, _ in ranked[:k]]

    # Calculate cost savings
    total = len(candidate_summaries)
    survivors = len(top_k)
    cost_reduction = ((total - survivors) / total * 100) if total > 0 else 100.0

    return SieveResult(
        ranked_candidates=ranked,
        top_k_indices=top_k,
        total_candidates=total,
        survivors_count=survivors,
        cost_reduction_percent=round(cost_reduction, 1),
    )
