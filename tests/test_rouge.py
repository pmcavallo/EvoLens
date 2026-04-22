"""Sanity-check tests for the ROUGE implementation.

Tests cover ROUGE-1, ROUGE-2, and ROUGE-L with focus on recall behavior
and the LCS mechanic.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolens.calibration.rouge import compute_rouge


class TestRougeN:
    """ROUGE-N recall captures content coverage."""

    def test_perfect_recall_with_extra_content(self):
        """Candidate contains all reference words plus more — recall should be 1.0."""
        result = compute_rouge(
            candidate="The Fed raised interest rates by 0.25 percent this morning signaling hawkish stance",
            reference="The Fed raised interest rates by 0.25 percent this morning",
        )
        assert result["rouge1"]["recall"] == 1.0

    def test_low_recall_when_content_missing(self):
        """Candidate omits key content — recall should drop."""
        result = compute_rouge(
            candidate="The Fed raised rates this morning",
            reference="The Fed raised interest rates by 0.25 percent this morning",
        )
        assert result["rouge1"]["recall"] < 0.7

    def test_rouge2_captures_bigram_fluency(self):
        """ROUGE-2 distinguishes fluent from scrambled candidates."""
        ref = "The cat sat on the mat"
        # Fluent (preserves bigrams)
        fluent = compute_rouge(candidate="The cat sat on the mat today", reference=ref)
        # Scrambled (breaks bigrams)
        scrambled = compute_rouge(candidate="mat the on cat the sat", reference=ref)
        assert fluent["rouge2"]["f1"] > scrambled["rouge2"]["f1"]


class TestRougeL:
    """ROUGE-L captures sentence-level structure via LCS."""

    def test_insertion_tolerance(self):
        """ROUGE-L should tolerate insertions better than ROUGE-2."""
        ref = "The cat sat on the mat"
        cand = "The cat sat quickly on the mat"
        result = compute_rouge(candidate=cand, reference=ref)
        # ROUGE-L should be higher than ROUGE-2 because LCS handles insertion
        assert result["rougeL"]["recall"] >= result["rouge2"]["recall"]

    def test_lcs_captures_order(self):
        """Two candidates with same words but different order get different ROUGE-L."""
        ref = "The company reported strong quarterly results"
        ordered = compute_rouge(candidate="The company reported strong results", reference=ref)
        reversed_cand = compute_rouge(candidate="results strong reported company The", reference=ref)
        assert ordered["rougeL"]["f1"] > reversed_cand["rougeL"]["f1"]


class TestRecallVsPrecision:
    """ROUGE reports both precision and recall; recall is the defining property."""

    def test_verbose_candidate_high_recall_low_precision(self):
        """A very long candidate covering all reference content: high recall, low precision."""
        result = compute_rouge(
            candidate="Revenue increased by 12 percent in the third quarter driven by markets and launches across segments",
            reference="Revenue increased by 12 percent in the third quarter",
        )
        assert result["rouge1"]["recall"] > 0.8
        assert result["rouge1"]["precision"] < result["rouge1"]["recall"]

    def test_terse_candidate_high_precision_low_recall(self):
        """A very short candidate with correct words: high precision, low recall."""
        result = compute_rouge(
            candidate="Revenue increased",
            reference="Revenue increased by 12 percent in the third quarter",
        )
        assert result["rouge1"]["precision"] == 1.0
        assert result["rouge1"]["recall"] < 0.3


class TestEdgeCases:
    def test_identical_texts(self):
        text = "The board approved the merger unanimously"
        result = compute_rouge(candidate=text, reference=text)
        assert result["rouge1"]["f1"] == 1.0
        assert result["rouge2"]["f1"] == 1.0
        assert result["rougeL"]["f1"] == 1.0

    def test_no_overlap(self):
        result = compute_rouge(
            candidate="purple freedom silently",
            reference="quarterly earnings declined significantly",
        )
        assert result["rouge1"]["f1"] == 0.0
        assert result["rouge2"]["f1"] == 0.0

    @pytest.mark.parametrize(
        "candidate,reference",
        [
            ("the cat sat on the mat", "the cat is on the mat"),
            ("revenue grew quickly", "revenue increased by 12 percent"),
        ],
    )
    def test_scores_in_range(self, candidate, reference):
        result = compute_rouge(candidate, reference)
        for key in ["rouge1", "rouge2", "rougeL"]:
            for metric in ["precision", "recall", "f1"]:
                assert 0.0 <= result[key][metric] <= 1.0
