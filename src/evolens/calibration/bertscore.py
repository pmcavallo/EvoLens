"""BERTScore — Calibration pillar.

Semantic similarity metric using contextual embeddings from BERT.
Replaces n-gram matching with cosine similarity between token embeddings,
catching paraphrase that surface metrics miss.

Uses greedy matching: each token picks its best match independently.
Supports baseline rescaling to map the natural ~0.85 floor to 0.0.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _cosine_similarity_matrix(
    a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Compute pairwise cosine similarity between two sets of vectors.

    Args:
        a: Shape (m, d) — candidate token embeddings.
        b: Shape (n, d) — reference token embeddings.

    Returns:
        Shape (m, n) cosine similarity matrix.
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


def compute_bertscore(
    candidate_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    baseline: Optional[float] = None,
) -> dict[str, float]:
    """Compute BERTScore from pre-computed token embeddings.

    Args:
        candidate_embeddings: Shape (m, d) contextual embeddings for candidate tokens.
        reference_embeddings: Shape (n, d) contextual embeddings for reference tokens.
        baseline: If provided, rescale scores: (raw - baseline) / (1 - baseline).
                  For BERT-base English, baseline is approximately 0.85.

    Returns:
        Dictionary with 'precision', 'recall', 'f1', and if baseline is set,
        'precision_rescaled', 'recall_rescaled', 'f1_rescaled'.
    """
    sim_matrix = _cosine_similarity_matrix(candidate_embeddings, reference_embeddings)

    # Greedy matching: each token picks its best match independently
    # Precision: for each candidate token, max similarity against any reference token
    precision = float(np.mean(np.max(sim_matrix, axis=1)))

    # Recall: for each reference token, max similarity against any candidate token
    recall = float(np.mean(np.max(sim_matrix, axis=0)))

    # F1: harmonic mean
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    if baseline is not None:
        denom = 1.0 - baseline
        if denom > 0:
            result["precision_rescaled"] = (precision - baseline) / denom
            result["recall_rescaled"] = (recall - baseline) / denom
            result["f1_rescaled"] = (f1 - baseline) / denom
        else:
            result["precision_rescaled"] = 0.0
            result["recall_rescaled"] = 0.0
            result["f1_rescaled"] = 0.0

    return result


def compute_bertscore_from_text(
    candidate: str,
    reference: str,
    model_name: str = "bert-base-uncased",
    baseline: Optional[float] = 0.85,
) -> dict[str, float]:
    """Compute BERTScore from raw text using a transformer model.

    Requires torch and transformers to be installed.

    Args:
        candidate: The generated text to evaluate.
        reference: The human reference text.
        model_name: HuggingFace model name for contextual embeddings.
        baseline: Baseline for rescaling (default 0.85 for bert-base-uncased English).

    Returns:
        Dictionary with raw and rescaled precision, recall, f1.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "BERTScore from text requires torch and transformers. "
            "Install with: pip install torch transformers"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    with torch.no_grad():
        cand_inputs = tokenizer(candidate, return_tensors="pt", padding=True, truncation=True)
        ref_inputs = tokenizer(reference, return_tensors="pt", padding=True, truncation=True)

        cand_outputs = model(**cand_inputs)
        ref_outputs = model(**ref_inputs)

        # Use last hidden state, skip [CLS] and [SEP] tokens
        cand_embeds = cand_outputs.last_hidden_state[0, 1:-1, :].numpy()
        ref_embeds = ref_outputs.last_hidden_state[0, 1:-1, :].numpy()

    return compute_bertscore(cand_embeds, ref_embeds, baseline=baseline)
