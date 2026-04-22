"""Perplexity — Calibration-adjacent, primarily Stability pillar.

Reference-free metric measuring how predictable text is to a language model.
Perplexity = exp(-1/N * sum(log P(token_i | context))).

Lower perplexity means the model is less surprised by the text.
Does NOT measure correctness — a fluent hallucination scores low perplexity.

Primary use: stability monitoring (drift detection over time).
Single-reading interpretation: calibration-adjacent (model confidence).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence


def compute_perplexity_from_logprobs(
    log_probs: Sequence[float],
) -> dict[str, float]:
    """Compute perplexity from a sequence of log-probabilities.

    Args:
        log_probs: Log-probabilities (base e) assigned by the model to each
                   token in the sequence, conditioned on previous tokens.

    Returns:
        Dictionary with 'perplexity', 'avg_log_prob', and 'num_tokens'.
    """
    n = len(log_probs)
    if n == 0:
        return {"perplexity": float("inf"), "avg_log_prob": float("-inf"), "num_tokens": 0}

    avg_log_prob = sum(log_probs) / n
    perplexity = math.exp(-avg_log_prob)

    return {
        "perplexity": perplexity,
        "avg_log_prob": avg_log_prob,
        "num_tokens": n,
    }


def compute_perplexity(
    text: str,
    model_name: str = "gpt2",
    stride: Optional[int] = None,
) -> dict[str, float]:
    """Compute perplexity of text using a causal language model.

    Requires torch and transformers to be installed.

    Args:
        text: The text to evaluate.
        model_name: HuggingFace causal LM model name (default: gpt2).
        stride: Sliding window stride for long texts. If None, processes
                the full text at once (limited by model's max context length).

    Returns:
        Dictionary with 'perplexity', 'avg_log_prob', and 'num_tokens'.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Perplexity computation requires torch and transformers. "
            "Install with: pip install torch transformers"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    seq_len = input_ids.size(1)

    if seq_len <= 1:
        return {"perplexity": float("inf"), "avg_log_prob": float("-inf"), "num_tokens": 0}

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # Cross-entropy loss is the negative average log-probability
        neg_avg_log_prob = outputs.loss.item()

    perplexity = math.exp(neg_avg_log_prob)

    return {
        "perplexity": perplexity,
        "avg_log_prob": -neg_avg_log_prob,
        "num_tokens": seq_len - 1,  # N-1 predictions for N tokens
    }
