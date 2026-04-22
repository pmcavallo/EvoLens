# Discrimination — planned for v2

This package is a placeholder. It exists to signal that the three-pillar validation framework is the structural commitment of this project, not a rhetorical flourish.

## Pillar question

*Can the evaluator rank outputs by quality?*

In credit risk model validation, Discrimination answers whether the model separates good credits from bad. The instruments are Somers' D, KS statistic, Gini coefficient, and AUC.

In LLM evaluation, the same question is being rebuilt under different names: LLM-as-judge, Elo rating systems, pairwise preference tournaments, arena-style evaluations. The shared mechanic is ranking: the evaluator is asked to produce an ordering over outputs, and the system is measured by how well that ordering matches a reference ordering (human labels, expert judgments, or a stronger model's preferences).

## Metrics planned for v2

- **LLM-as-judge** — prompt a stronger model to score candidates; measure agreement with human raters
- **Pairwise preference win-rate** — head-to-head comparison between candidates
- **Elo / Bradley-Terry ratings** — aggregated from pairwise judgments across many pairs
- **Rank correlation** (Spearman, Kendall's tau) — between LLM-judge ordering and human ordering

## Why not v1

Discrimination metrics all require either a labeled dataset of quality rankings or a stronger judge model. Both push beyond the single-reference, zero-API-key design of EvoLens v1. Shipping a half-built Discrimination pillar would have been worse than shipping Calibration cleanly and declaring the framework.
