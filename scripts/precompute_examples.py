"""Pre-compute metric scores for the EvoLens synthetic example pairs.

Two-tier strategy:

* BLEU and ROUGE are computed live from the EvoLens Python reference
  implementations in ``src/evolens/calibration/``. These are pure-Python
  and require only the standard library, so the script runs anywhere.

* BERTScore and Perplexity are embedded as curated values on each example.
  Computing them live would require torch + transformers + bert-score plus
  a ~400 MB BERT checkpoint and GPT-2 weights, which is too heavy for a
  static-site build step. The embedded values were selected so that each
  example makes its intended pedagogical point (paraphrase-penalty,
  hallucination vs. under-coverage asymmetry, baseline rescaling, fluent-
  but-wrong confidence, and so on) and are consistent with what bert-score
  and GPT-2 produce on texts of this shape.

Running the script regenerates
``dashboard/src/content/examples/synthetic-examples.json`` deterministically.

Usage::

    py scripts/precompute_examples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolens.calibration.bleu import compute_bleu
from evolens.calibration.rouge import compute_rouge


EXAMPLES = [
    {
        "id": 1,
        "name": "Near-identical paraphrase",
        "description": "Same meaning expressed with different words. BLEU low (surface mismatch), BERTScore high (semantic match). Teaches the paraphrase-penalty problem.",
        "scenario": "paraphrase",
        "reference": "The board approved the merger unanimously after reviewing the financial projections.",
        "candidate": "The directors gave their full consent to the acquisition following an examination of the fiscal forecasts.",
        "bertscore": {
            "precision": 0.92, "recall": 0.91, "f1": 0.915,
            "precision_rescaled": 0.467, "recall_rescaled": 0.400, "f1_rescaled": 0.433,
        },
        "perplexity": {"perplexity": 18.3, "avg_log_prob": -2.907, "num_tokens": 15},
    },
    {
        "id": 2,
        "name": "Word-order rearrangement",
        "description": "Identical vocabulary, different word order. BLEU moderate (bigram breaks), BERTScore very high (same meaning). The baseline where surface and semantic metrics agree.",
        "scenario": "reorder",
        "reference": "The cat sat on the mat near the window.",
        "candidate": "Near the window the cat sat on the mat.",
        "bertscore": {
            "precision": 0.97, "recall": 0.97, "f1": 0.970,
            "precision_rescaled": 0.800, "recall_rescaled": 0.800, "f1_rescaled": 0.800,
        },
        "perplexity": {"perplexity": 38.7, "avg_log_prob": -3.656, "num_tokens": 9},
    },
    {
        "id": 3,
        "name": "Hallucination case",
        "description": "Candidate adds content not in the reference. BERTScore Precision drops (hallucinated words have no reference match), Recall stays high (reference content is covered).",
        "scenario": "hallucination",
        "reference": "Revenue increased by 12 percent in the third quarter.",
        "candidate": "Revenue increased by 12 percent in the third quarter, driven primarily by strong performance in the Asian markets and new product launches across all segments.",
        "bertscore": {
            "precision": 0.88, "recall": 0.96, "f1": 0.918,
            "precision_rescaled": 0.200, "recall_rescaled": 0.733, "f1_rescaled": 0.453,
        },
        "perplexity": {"perplexity": 14.7, "avg_log_prob": -2.688, "num_tokens": 24},
    },
    {
        "id": 4,
        "name": "Under-coverage case",
        "description": "Candidate omits key reference content. BERTScore Recall drops (reference words unmatched), Precision stays high (candidate content is accurate).",
        "scenario": "undercoverage",
        "reference": "The Federal Reserve raised interest rates by 25 basis points this morning, citing persistent inflation in the housing and services sectors.",
        "candidate": "The Federal Reserve raised interest rates this morning.",
        "bertscore": {
            "precision": 0.96, "recall": 0.87, "f1": 0.913,
            "precision_rescaled": 0.733, "recall_rescaled": 0.133, "f1_rescaled": 0.420,
        },
        "perplexity": {"perplexity": 11.2, "avg_log_prob": -2.416, "num_tokens": 8},
    },
    {
        "id": 5,
        "name": "Completely unrelated sentences",
        "description": "No content relationship. Raw BERTScore ~0.85, rescaled ~0.00. Teaches why baseline rescaling exists — without it, 0.85 looks like a strong match.",
        "scenario": "unrelated",
        "reference": "The quarterly earnings report showed a significant decline in operating margins.",
        "candidate": "The golden retriever chased butterflies through the meadow on a sunny afternoon.",
        "bertscore": {
            "precision": 0.855, "recall": 0.852, "f1": 0.854,
            "precision_rescaled": 0.033, "recall_rescaled": 0.013, "f1_rescaled": 0.027,
        },
        "perplexity": {"perplexity": 22.1, "avg_log_prob": -3.096, "num_tokens": 10},
    },
    {
        "id": 6,
        "name": "Fluent but wrong",
        "description": "Low perplexity (model is confident) but factually incorrect. Teaches why perplexity alone is dangerous — a fluent hallucination scores well on confidence while being wrong.",
        "scenario": "fluent_wrong",
        "reference": "The United States purchased Alaska from Russia in 1867 for $7.2 million.",
        "candidate": "The United States purchased Alaska from Russia in 1853 for $7.2 million, in a transaction negotiated by Secretary of State Daniel Webster.",
        "bertscore": {
            "precision": 0.93, "recall": 0.95, "f1": 0.940,
            "precision_rescaled": 0.533, "recall_rescaled": 0.667, "f1_rescaled": 0.600,
        },
        "perplexity": {"perplexity": 9.8, "avg_log_prob": -2.282, "num_tokens": 19},
    },
    {
        "id": 7,
        "name": "Summary captures gist but loses specifics",
        "description": "ROUGE-1 decent (key words present), ROUGE-L lower (sentence structure diverges). Teaches the LCS mechanic and how different ROUGE variants capture different properties.",
        "scenario": "gist_summary",
        "reference": "The company reported a 15 percent increase in annual revenue, reaching $4.2 billion, while net profit margins expanded to 18 percent from 14 percent the previous year.",
        "candidate": "Annual revenue rose 15 percent to $4.2 billion. Profit margins improved significantly year over year.",
        "bertscore": {
            "precision": 0.92, "recall": 0.89, "f1": 0.905,
            "precision_rescaled": 0.467, "recall_rescaled": 0.267, "f1_rescaled": 0.367,
        },
        "perplexity": {"perplexity": 16.4, "avg_log_prob": -2.797, "num_tokens": 12},
    },
    {
        "id": 8,
        "name": "Mixed quality (one good sentence, one bad)",
        "description": "Multi-sentence output where the first sentence is accurate and the second is fabricated. All metrics show middling scores, teaching the importance of aggregation discipline.",
        "scenario": "mixed_quality",
        "reference": "Global temperatures rose by 1.1 degrees Celsius above pre-industrial levels. The Paris Agreement aims to limit warming to 1.5 degrees.",
        "candidate": "Global temperatures rose by 1.1 degrees Celsius above pre-industrial levels. Scientists predict that all major cities will be underwater by 2050.",
        "bertscore": {
            "precision": 0.90, "recall": 0.89, "f1": 0.895,
            "precision_rescaled": 0.333, "recall_rescaled": 0.267, "f1_rescaled": 0.300,
        },
        "perplexity": {"perplexity": 19.5, "avg_log_prob": -2.970, "num_tokens": 18},
    },
]


def _round3(d: dict) -> dict:
    """Round every float value to 3 decimal places."""
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()}


def main() -> None:
    """Compute BLEU/ROUGE for all examples and write the merged JSON."""
    results = []

    for ex in EXAMPLES:
        ref = ex["reference"]
        cand = ex["candidate"]

        bleu = compute_bleu(cand, ref)
        rouge = compute_rouge(cand, ref)

        entry = {
            "id": ex["id"],
            "name": ex["name"],
            "description": ex["description"],
            "scenario": ex["scenario"],
            "reference": ref,
            "candidate": cand,
            "scores": {
                "bleu": _round3(bleu),
                "rouge1": _round3(rouge["rouge1"]),
                "rouge2": _round3(rouge["rouge2"]),
                "rougeL": _round3(rouge["rougeL"]),
                "bertscore": ex["bertscore"],
                "perplexity": ex["perplexity"],
            },
        }
        results.append(entry)

        print(f"Example {ex['id']}: {ex['name']}")
        print(f"  BLEU: {bleu['bleu']:.3f}")
        print(f"  ROUGE-1 F1: {rouge['rouge1']['f1']:.3f}")
        print(f"  ROUGE-2 F1: {rouge['rouge2']['f1']:.3f}")
        print(f"  ROUGE-L F1: {rouge['rougeL']['f1']:.3f}")
        print(f"  BERTScore F1 (rescaled): {ex['bertscore']['f1_rescaled']:.3f}  [curated]")
        print(f"  Perplexity: {ex['perplexity']['perplexity']:.1f}  [curated]")
        print()

    out_path = (
        Path(__file__).resolve().parent.parent
        / "dashboard" / "src" / "content" / "examples" / "synthetic-examples.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(results)} examples to {out_path}")


if __name__ == "__main__":
    main()
