"""Sanity-check tests for the perplexity implementation.

Tests the core math (log-prob to perplexity conversion) using synthetic
log-probabilities. Does not require torch or transformers.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolens.calibration.perplexity import compute_perplexity_from_logprobs


class TestBasicComputation:
    """Core perplexity formula: exp(-1/N * sum(log_probs))."""

    def test_uniform_distribution(self):
        """If model assigns 1/10 probability at each step, perplexity should be 10."""
        log_probs = [math.log(0.1)] * 10  # uniform p=0.1
        result = compute_perplexity_from_logprobs(log_probs)
        assert abs(result["perplexity"] - 10.0) < 0.01

    def test_certain_model(self):
        """If model assigns probability 1.0 at each step, perplexity should be 1."""
        log_probs = [math.log(1.0)] * 5  # p=1.0
        result = compute_perplexity_from_logprobs(log_probs)
        assert abs(result["perplexity"] - 1.0) < 0.01

    def test_binary_choice(self):
        """If model assigns 0.5 at each step, perplexity should be 2."""
        log_probs = [math.log(0.5)] * 8
        result = compute_perplexity_from_logprobs(log_probs)
        assert abs(result["perplexity"] - 2.0) < 0.01

    def test_mixed_probabilities(self):
        """Manual computation check."""
        log_probs = [math.log(0.8), math.log(0.2), math.log(0.5)]
        avg_log_prob = sum(log_probs) / 3
        expected_ppl = math.exp(-avg_log_prob)
        result = compute_perplexity_from_logprobs(log_probs)
        assert abs(result["perplexity"] - expected_ppl) < 0.01


class TestInverseRelationship:
    """Perplexity is inversely related to probability."""

    def test_higher_probs_mean_lower_perplexity(self):
        """More confident predictions should produce lower perplexity."""
        confident = compute_perplexity_from_logprobs([math.log(0.9)] * 10)
        uncertain = compute_perplexity_from_logprobs([math.log(0.1)] * 10)
        assert confident["perplexity"] < uncertain["perplexity"]

    def test_monotonic_relationship(self):
        """Increasing probability should monotonically decrease perplexity."""
        probs = [0.1, 0.3, 0.5, 0.7, 0.9]
        perplexities = []
        for p in probs:
            result = compute_perplexity_from_logprobs([math.log(p)] * 5)
            perplexities.append(result["perplexity"])
        for i in range(len(perplexities) - 1):
            assert perplexities[i] > perplexities[i + 1]


class TestLengthNormalization:
    """Perplexity should normalize by sequence length."""

    def test_same_perplexity_regardless_of_length(self):
        """Same per-token probability, different lengths, same perplexity."""
        short = compute_perplexity_from_logprobs([math.log(0.3)] * 5)
        long = compute_perplexity_from_logprobs([math.log(0.3)] * 50)
        assert abs(short["perplexity"] - long["perplexity"]) < 0.01


class TestEdgeCases:
    def test_empty_sequence(self):
        result = compute_perplexity_from_logprobs([])
        assert result["perplexity"] == float("inf")
        assert result["num_tokens"] == 0

    def test_single_token(self):
        result = compute_perplexity_from_logprobs([math.log(0.5)])
        assert abs(result["perplexity"] - 2.0) < 0.01
        assert result["num_tokens"] == 1

    def test_num_tokens_reported(self):
        result = compute_perplexity_from_logprobs([math.log(0.5)] * 7)
        assert result["num_tokens"] == 7


class TestBranchingFactorIntuition:
    """Perplexity equals the effective branching factor."""

    @pytest.mark.parametrize("branching_factor", [2, 5, 10, 50, 100])
    def test_branching_factor(self, branching_factor):
        """Uniform p = 1/k should give perplexity = k."""
        p = 1.0 / branching_factor
        result = compute_perplexity_from_logprobs([math.log(p)] * 20)
        assert abs(result["perplexity"] - branching_factor) < 0.1
