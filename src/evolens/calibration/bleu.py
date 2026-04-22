"""BLEU (Bilingual Evaluation Understudy) — Calibration pillar.

Reference-based metric measuring n-gram precision between candidate and reference.
Four defenses: clipped precision, n-gram expansion (1-4), brevity penalty, geometric mean.

Implements smoothed sentence-level BLEU following Chen & Cherry (2014) method 1
(add-epsilon smoothing) to handle zero n-gram counts on short texts.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenization."""
    return text.lower().split()


def _count_ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    """Count n-grams in a token sequence."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _clipped_precision(
    candidate_tokens: Sequence[str],
    reference_tokens: Sequence[str],
    n: int,
) -> tuple[int, int]:
    """Compute clipped n-gram precision counts.

    Returns (clipped_matches, total_candidate_ngrams).
    """
    cand_ngrams = _count_ngrams(candidate_tokens, n)
    ref_ngrams = _count_ngrams(reference_tokens, n)

    clipped = 0
    for ngram, count in cand_ngrams.items():
        clipped += min(count, ref_ngrams.get(ngram, 0))

    total = max(sum(cand_ngrams.values()), 0)
    return clipped, total


def compute_bleu(
    candidate: str,
    reference: str,
    max_n: int = 4,
    smoothing: bool = True,
    epsilon: float = 0.1,
) -> dict[str, float]:
    """Compute sentence-level BLEU score.

    Args:
        candidate: The generated text to evaluate.
        reference: The human reference text.
        max_n: Maximum n-gram order (default 4 for standard BLEU-4).
        smoothing: Apply add-epsilon smoothing for zero counts.
        epsilon: Smoothing constant added to zero-count numerators.

    Returns:
        Dictionary with 'bleu' (final score), 'brevity_penalty', and
        per-level precisions 'p1' through 'p{max_n}'.
    """
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)

    c = len(cand_tokens)
    r = len(ref_tokens)

    if c == 0:
        return {
            "bleu": 0.0,
            "brevity_penalty": 0.0,
            **{f"p{n}": 0.0 for n in range(1, max_n + 1)},
        }

    # Brevity penalty: asymmetric, only fires when candidate is shorter
    if c > r:
        bp = 1.0
    else:
        bp = math.exp(1 - r / c)

    # Clipped n-gram precisions with optional smoothing
    log_precisions: list[float] = []
    precisions: dict[str, float] = {}

    for n in range(1, max_n + 1):
        matches, total = _clipped_precision(cand_tokens, ref_tokens, n)

        if total == 0:
            precisions[f"p{n}"] = 0.0
            log_precisions.append(float("-inf"))
            continue

        if matches == 0 and smoothing:
            matches_adj = epsilon
        else:
            matches_adj = float(matches)

        p = matches_adj / total
        precisions[f"p{n}"] = p
        log_precisions.append(math.log(p) if p > 0 else float("-inf"))

    # Geometric mean in log space (equal weights)
    if any(lp == float("-inf") for lp in log_precisions):
        bleu = 0.0
    else:
        weights = [1.0 / max_n] * max_n
        log_avg = sum(w * lp for w, lp in zip(weights, log_precisions))
        bleu = bp * math.exp(log_avg)

    return {"bleu": bleu, "brevity_penalty": bp, **precisions}
