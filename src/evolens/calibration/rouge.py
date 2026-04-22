"""ROUGE (Recall-Oriented Understudy for Gisting Evaluation) — Calibration pillar.

Reference-based metric family measuring recall of n-grams and subsequences.
Implements ROUGE-1, ROUGE-2, and ROUGE-L.

Key inversion from BLEU: denominator is reference length (recall) not
candidate length (precision), because summarization's failure mode is omission.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenization."""
    return text.lower().split()


def _count_ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    """Count n-grams in a token sequence."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _rouge_n(
    candidate_tokens: Sequence[str],
    reference_tokens: Sequence[str],
    n: int,
) -> dict[str, float]:
    """Compute ROUGE-N precision, recall, and F1."""
    cand_ngrams = _count_ngrams(candidate_tokens, n)
    ref_ngrams = _count_ngrams(reference_tokens, n)

    matches = 0
    for ngram, count in ref_ngrams.items():
        matches += min(count, cand_ngrams.get(ngram, 0))

    ref_total = max(sum(ref_ngrams.values()), 1)
    cand_total = max(sum(cand_ngrams.values()), 1)

    recall = matches / ref_total
    precision = matches / cand_total

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Compute length of the longest common subsequence."""
    m, n = len(a), len(b)
    # Space-optimized: only need two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


def _rouge_l(
    candidate_tokens: Sequence[str],
    reference_tokens: Sequence[str],
) -> dict[str, float]:
    """Compute ROUGE-L using longest common subsequence."""
    lcs = _lcs_length(candidate_tokens, reference_tokens)

    ref_len = max(len(reference_tokens), 1)
    cand_len = max(len(candidate_tokens), 1)

    recall = lcs / ref_len
    precision = lcs / cand_len

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_rouge(
    candidate: str,
    reference: str,
) -> dict[str, dict[str, float]]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Args:
        candidate: The generated text to evaluate.
        reference: The human reference text.

    Returns:
        Dictionary with keys 'rouge1', 'rouge2', 'rougeL', each containing
        'precision', 'recall', and 'f1'.
    """
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)

    return {
        "rouge1": _rouge_n(cand_tokens, ref_tokens, 1),
        "rouge2": _rouge_n(cand_tokens, ref_tokens, 2),
        "rougeL": _rouge_l(cand_tokens, ref_tokens),
    }
