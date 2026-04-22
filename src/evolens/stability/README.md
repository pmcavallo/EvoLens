# Stability — planned for v3

This package is a placeholder. It exists to signal that the three-pillar validation framework is the structural commitment of this project, not a rhetorical flourish.

## Pillar question

*Is model behavior drifting over time?*

In credit risk model validation, Stability answers whether the scored population has shifted since the model was built. The canonical instrument is PSI (Population Stability Index), computed on the distribution of risk scores or feature values between a reference period and a monitoring period.

In LLM evaluation, drift can appear in at least three places:

1. **Input drift** — the prompts the model is seeing have shifted (new vocabulary, new topics, new styles).
2. **Output drift** — the model's responses have shifted at fixed prompts (e.g. after a provider model update).
3. **Quality drift** — the same reference-based calibration metrics (BLEU, BERTScore, etc.) are trending downward on a stable benchmark.

Perplexity, which is implemented in the Calibration pillar of v1, is the cleanest single-metric bridge to Stability: perplexity of the current model on a fixed canary set, tracked over time, gives an early warning signal for output drift even when the model has no reference to compare against.

## Metrics planned for v3

- **Perplexity drift** on a fixed canary set
- **Embedding drift** — cosine distance of output embeddings against a reference distribution
- **PSI over bucketed metric scores** — apply the credit-risk PSI mechanic directly to BLEU/BERTScore distributions between a reference window and a monitoring window
- **Distribution divergence** (KL, Jensen-Shannon) between reference and current output embedding distributions

## Why not v1

Stability is a rolling-window concept. It requires a reference time period, a monitoring time period, and a mechanism for persisting scored output over time. A static single-page dashboard cannot demonstrate it honestly. The Stability pillar will come back into scope when EvoLens grows into a monitoring workflow rather than a one-shot scoring tool.
