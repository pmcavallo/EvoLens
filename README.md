# EvoLens

**Evaluation instrument for the Calibration pillar of AI model validation.**

EvoLens computes BLEU, ROUGE, BERTScore, and Perplexity on LLM outputs against reference texts, and explains what each metric measures in the language of model risk management. Every score is clickable and expands an inline pedagogy panel written for reviewers, not research papers.

> **Live demo:** [pmcavallo.github.io/EvoLens/](https://pmcavallo.github.io/EvoLens/)
>
> The demo ships with eight pre-computed synthetic example pairs. No API keys required. A power-user "bring your own API key" flow is available for evaluating outputs from OpenAI or Anthropic against your own references.

---

## The three-pillar validation framework

EvoLens is framed through the three-pillar validation framework borrowed from SR 11-7 credit risk model validation. Under SR 11-7, a credit risk model is not validated by one number — it is validated against three distinct questions. LLM evaluation is rebuilding the same framework.

| Pillar | Credit risk question | LLM equivalent | Metrics in this pillar |
|---|---|---|---|
| **Discrimination** | Does the model separate good from bad? | Does the evaluator rank outputs by quality? | LLM-as-judge, Elo, pairwise preference |
| **Calibration** | Do predictions match observations? | Does the candidate match the reference? | **BLEU, ROUGE, BERTScore, Perplexity** |
| **Stability** | Is the scored population drifting? | Is model behavior drifting over time? | Perplexity drift, embedding drift |

The four metrics implemented in EvoLens v1 all sit in the **Calibration** pillar. Perplexity is the deliberate edge case: at a single point in time it is a calibration-adjacent measurement of model confidence, but tracked over time it becomes the clearest Stability signal. The tool surfaces that distinction explicitly.

**Discrimination** and **Stability** are planned extensions for v2 and v3. The repository structure reserves space for them (`src/evolens/discrimination/`, `src/evolens/stability/`) to signal that the framework is the product, not any single metric.

---

## Why this exists

The three-pillar validation framework — Discrimination, Calibration, Stability — has organized credit risk model validation for decades. Every validator who works under SR 11-7 knows the shape of it: a model is not trusted because of one number, but because it answers three structurally different questions. In parallel, and largely independently, LLM evaluation has been rebuilding the same framework under different names. LLM-as-judge and pairwise preference are Discrimination. Reference-based metrics like BLEU, ROUGE, and BERTScore are Calibration. Perplexity drift and embedding drift are Stability.

The mapping between the two has not been systematically documented in existing public work. Tutorials treat BLEU, ROUGE, BERTScore, and perplexity as a flat list of metrics. Regulator-facing material treats the three pillars as a credit-risk-only framework. The two conversations have not met.

EvoLens is that mapping, delivered as a working tool. The four metrics implemented in v1 are the Calibration pillar, each explained in the language of model risk management. The folder structure reserves space for Discrimination and Stability so the framework is legible from the code itself.

---

## Quick start

### Run the dashboard locally

```bash
cd dashboard
npm install
npm run dev
```

Open the printed URL (default `http://localhost:5173/EvoLens/`). Select one of the eight example scenarios, click any metric score to expand its pedagogy panel, or switch to "Evaluate your own" to paste a reference/candidate pair or generate one with an API key.

### Run the Python metric tests

```bash
py -m pytest tests/ -q
```

All 52 tests exercise the internal mechanics of each metric (clipped precision, brevity penalty, LCS, cosine similarity, greedy matching, baseline rescaling, branching factor) and should pass in under a second.

### Regenerate the pre-computed examples

```bash
py scripts/precompute_examples.py
```

BLEU and ROUGE are recomputed from the Python reference implementations in `src/evolens/calibration/`. BERTScore and Perplexity values are curated per-example to illustrate the intended pedagogical point for each scenario (paraphrase penalty, hallucination asymmetry, baseline rescaling, fluent-but-wrong, etc.). See the script docstring for detail.

---

## Repository layout

```
llm-evaluation-metrics/
├── README.md                       ← this file
├── src/evolens/
│   ├── calibration/                 ← BLEU / ROUGE / BERTScore / Perplexity
│   ├── discrimination/              ← placeholder (v2 scope)
│   └── stability/                   ← placeholder (v3 scope)
├── tests/                           52 Python unit tests
├── scripts/
│   └── precompute_examples.py       regenerates synthetic-examples.json
└── dashboard/                       React + Vite + Tailwind single-page app
    └── src/content/
        ├── pedagogy/*.json          per-metric pedagogy copy
        └── examples/*.json          pre-computed example pairs
```

---

## The integrated pedagogy layer

Every metric card in the dashboard is clickable. Expanding a card reveals:

1. **What this number measures** — one plain-English sentence, no jargon.
2. **Why it measures that** — the design-follows-failure-mode reasoning for each metric.
3. **Pillar badge** — labeled *Calibration*, itself clickable to reveal the three-pillar overview.
4. **What a regulator or MRM reviewer would ask** — three to five pointed questions per metric.
5. **Interpretation guide** — what "good" and "bad" look like for this metric, with the caveat that thresholds are task-dependent.
6. **Credit risk parallel** — the one-line mapping back to an SR 11-7 analogue.
7. **Learn more** — an expandable deeper-dive section with the full explanation, history, and mechanics of the metric.

The pedagogy strings are stored as JSON in `dashboard/src/content/pedagogy/` rather than inlined in React components, so the same content can be reused in newsletter issues, LinkedIn posts, or future projects without rewriting.

---

## What is out of scope for v1

- Drift detection and rolling-window analysis
- Threshold alerting
- Multi-LLM side-by-side comparison
- User authentication or accounts (static site — no users to authenticate)
- Server-side storage of user evaluations (data sovereignty is a feature)
- Full implementations of the Discrimination and Stability pillars (placeholders only for v1)

---

## Tech stack

| Layer | Choice |
|---|---|
| Python metric reference | pure standard library + numpy |
| Python tests | pytest |
| Frontend | React 19 + Vite + TypeScript |
| Styling | Tailwind CSS v4 |
| In-browser BLEU/ROUGE (BYO mode) | hand-ported TS mirrors of the Python impls |
| Deployment | GitHub Pages via GitHub Actions |

---

## Data disclosure

All example data shipped in this repository is synthetic. No real financial, client, or proprietary data is used anywhere in the project.

---

## Credit

Built by Paulo Cavallo, PhD. AI governance practitioner, senior credit risk model developer, and author of the *AI Under Audit* newsletter.

EvoLens is the Calibration-pillar instrument in a larger thesis: that regulated industries and the AI industry are converging on the same validation framework, and that people who can translate between the two have work to do.

Licensed under the MIT License — see [`LICENSE`](./LICENSE).
