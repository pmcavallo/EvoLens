"""Sanity-check tests for the BLEU implementation.

Tests cover the four defenses: clipped precision, n-gram expansion,
brevity penalty, and geometric mean behavior.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolens.calibration.bleu import compute_bleu


class TestClippedPrecision:
    """Clipping prevents repetition gaming."""

    def test_repetition_gaming_scores_low(self):
        result = compute_bleu(
            candidate="the the the the the the the",
            reference="the cat is on the mat",
        )
        # Unigram precision should be 2/7 (clipped), not 7/7
        assert result["p1"] < 0.35

    def test_normal_candidate_not_clipped(self):
        result = compute_bleu(
            candidate="the cat sat on the mat",
            reference="the cat is on the mat",
        )
        # 5 of 6 words match, none clipped
        assert abs(result["p1"] - 5 / 6) < 0.01


class TestBrevityPenalty:
    """Brevity penalty prevents too-short gaming."""

    def test_equal_length_no_penalty(self):
        result = compute_bleu(
            candidate="the cat sat on the mat",
            reference="the cat is on the mat",
        )
        assert result["brevity_penalty"] == 1.0

    def test_longer_candidate_no_penalty(self):
        result = compute_bleu(
            candidate="the cat sat on the mat today",
            reference="the cat is on the mat",
        )
        assert result["brevity_penalty"] == 1.0

    def test_shorter_candidate_penalized(self):
        result = compute_bleu(
            candidate="the mat",
            reference="the cat is on the mat",
        )
        assert result["brevity_penalty"] < 0.5

    def test_very_short_candidate_heavily_penalized(self):
        result = compute_bleu(
            candidate="the",
            reference="the cat is on the mat",
        )
        assert result["brevity_penalty"] < 0.2


class TestGeometricMean:
    """Geometric mean zeros out when any n-gram level fails."""

    def test_zero_4gram_zeros_score_without_smoothing(self):
        # Short text where 4-grams can't match
        result = compute_bleu(
            candidate="the cat sat on the mat",
            reference="the cat is on the mat",
            smoothing=False,
        )
        # p4=0 should zero the geometric mean
        assert result["bleu"] == 0.0

    def test_smoothing_prevents_zero(self):
        result = compute_bleu(
            candidate="the cat sat on the mat",
            reference="the cat is on the mat",
            smoothing=True,
        )
        # With smoothing, should be > 0
        assert result["bleu"] > 0.0


class TestEdgeCases:
    """Edge cases and boundary behavior."""

    def test_empty_candidate(self):
        result = compute_bleu(candidate="", reference="the cat is on the mat")
        assert result["bleu"] == 0.0

    def test_identical_texts(self):
        text = "the cat is on the mat near the window"
        result = compute_bleu(candidate=text, reference=text)
        assert result["bleu"] > 0.9
        assert result["brevity_penalty"] == 1.0

    def test_completely_different(self):
        result = compute_bleu(
            candidate="purple silently freedom underneath",
            reference="the quarterly report showed declining margins",
        )
        assert result["bleu"] < 0.05


class TestScoreRange:
    """BLEU scores should be between 0 and 1."""

    @pytest.mark.parametrize(
        "candidate,reference",
        [
            ("the cat sat on the mat", "the cat is on the mat"),
            ("revenue grew quickly", "revenue increased by 12 percent"),
            ("hello world", "goodbye moon"),
        ],
    )
    def test_score_in_range(self, candidate, reference):
        result = compute_bleu(candidate, reference)
        assert 0.0 <= result["bleu"] <= 1.0
        assert 0.0 <= result["brevity_penalty"] <= 1.0
        for n in range(1, 5):
            assert 0.0 <= result[f"p{n}"] <= 1.0
