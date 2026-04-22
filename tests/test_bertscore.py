"""Sanity-check tests for the BERTScore implementation.

Tests the core math (cosine similarity, greedy matching, rescaling) using
synthetic embeddings. Does not require torch or transformers.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolens.calibration.bertscore import compute_bertscore, _cosine_similarity_matrix


class TestCosineSimilarity:
    """Cosine similarity matrix computation."""

    def test_identical_vectors(self):
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0]])
        sim = _cosine_similarity_matrix(a, b)
        assert abs(sim[0, 0] - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[0.0, 1.0, 0.0]])
        sim = _cosine_similarity_matrix(a, b)
        assert abs(sim[0, 0]) < 1e-6

    def test_opposite_vectors(self):
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[-1.0, 0.0, 0.0]])
        sim = _cosine_similarity_matrix(a, b)
        assert abs(sim[0, 0] - (-1.0)) < 1e-6

    def test_matrix_shape(self):
        a = np.random.randn(5, 768)
        b = np.random.randn(7, 768)
        sim = _cosine_similarity_matrix(a, b)
        assert sim.shape == (5, 7)


class TestGreedyMatching:
    """Greedy matching: each token picks its best match independently."""

    def test_perfect_match(self):
        """Identical embeddings should give perfect scores."""
        embeddings = np.random.randn(5, 768)
        result = compute_bertscore(embeddings, embeddings)
        assert abs(result["precision"] - 1.0) < 1e-5
        assert abs(result["recall"] - 1.0) < 1e-5
        assert abs(result["f1"] - 1.0) < 1e-5

    def test_precision_recall_asymmetry(self):
        """More candidate tokens than reference: precision drops, recall stays high."""
        np.random.seed(42)
        # Reference: 3 tokens, Candidate: 6 tokens (3 matching + 3 random)
        ref = np.random.randn(3, 64)
        # First 3 candidate tokens match reference closely
        cand = np.vstack([ref + 0.01 * np.random.randn(3, 64), np.random.randn(3, 64)])
        result = compute_bertscore(cand, ref)
        # Recall should be high (reference tokens all matched)
        # Precision should be lower (extra candidate tokens poorly matched)
        assert result["recall"] > result["precision"]

    def test_recall_drops_with_missing_content(self):
        """Fewer candidate tokens than reference: recall drops."""
        np.random.seed(42)
        ref = np.random.randn(6, 64)
        # Candidate only covers first 2 reference tokens
        cand = ref[:2] + 0.01 * np.random.randn(2, 64)
        result = compute_bertscore(cand, ref)
        assert result["precision"] > result["recall"]


class TestBaselineRescaling:
    """Baseline rescaling maps the natural floor to 0."""

    def test_baseline_equals_raw_gives_zero(self):
        """If raw score equals baseline, rescaled should be ~0."""
        # Create embeddings that produce scores near 0.85
        np.random.seed(42)
        a = np.random.randn(5, 768)
        b = np.random.randn(5, 768)
        raw_result = compute_bertscore(a, b, baseline=None)
        # Rescale with baseline set to the raw f1
        result = compute_bertscore(a, b, baseline=raw_result["f1"])
        assert abs(result["f1_rescaled"]) < 0.01

    def test_perfect_score_rescales_to_one(self):
        embeddings = np.random.randn(5, 768)
        result = compute_bertscore(embeddings, embeddings, baseline=0.85)
        assert abs(result["f1_rescaled"] - 1.0) < 1e-5

    def test_rescaling_preserves_order(self):
        """Higher raw scores should produce higher rescaled scores."""
        np.random.seed(42)
        ref = np.random.randn(5, 64)
        # Close match
        close_cand = ref + 0.1 * np.random.randn(5, 64)
        # Far match
        far_cand = np.random.randn(5, 64)

        close_result = compute_bertscore(close_cand, ref, baseline=0.5)
        far_result = compute_bertscore(far_cand, ref, baseline=0.5)
        assert close_result["f1_rescaled"] > far_result["f1_rescaled"]


class TestF1HarmonicMean:
    """F1 should punish imbalance."""

    def test_balanced_scores(self):
        embeddings = np.random.randn(5, 768)
        result = compute_bertscore(embeddings, embeddings)
        # When P and R are equal, F1 should equal them
        assert abs(result["f1"] - result["precision"]) < 1e-5

    def test_imbalanced_scores_low_f1(self):
        """When precision and recall diverge, F1 should be closer to the lower one."""
        np.random.seed(42)
        ref = np.random.randn(10, 64)
        cand = np.vstack([ref[:2] + 0.01 * np.random.randn(2, 64), np.random.randn(8, 64)])
        result = compute_bertscore(cand, ref)
        avg = (result["precision"] + result["recall"]) / 2
        # Harmonic mean should be less than arithmetic mean when values diverge
        assert result["f1"] <= avg + 0.01
