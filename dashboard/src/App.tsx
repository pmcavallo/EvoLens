import { useState } from 'react';
import ExampleSelector from './components/ExampleSelector';
import MetricCard from './components/MetricCard';
import TextDisplay from './components/TextDisplay';
import BYOPanel from './components/BYOPanel';
import type { Example, ExampleScores } from './types';

import syntheticExamples from './content/examples/synthetic-examples.json';
import bleuPedagogy from './content/pedagogy/bleu.json';
import rougePedagogy from './content/pedagogy/rouge.json';
import bertscorePedagogy from './content/pedagogy/bertscore.json';
import perplexityPedagogy from './content/pedagogy/perplexity.json';

const examples = syntheticExamples as Example[];

export default function App() {
  const [selected, setSelected] = useState<Example>(examples[0]);
  const [showBYO, setShowBYO] = useState(false);
  const [byoScores, setByoScores] = useState<Partial<ExampleScores> | null>(null);
  const [byoReference, setByoReference] = useState('');
  const [byoCandidate, setByoCandidate] = useState('');

  const isPrecomputed = !showBYO || byoScores === null;
  const scores = isPrecomputed ? selected.scores : byoScores;
  const refText = isPrecomputed ? selected.reference : byoReference;
  const candText = isPrecomputed ? selected.candidate : byoCandidate;

  function handleBYOResults(
    newScores: Partial<ExampleScores>,
    reference: string,
    candidate: string
  ) {
    setByoScores(newScores);
    setByoReference(reference);
    setByoCandidate(candidate);
  }

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-navy text-white">
        <div className="max-w-5xl mx-auto px-6 py-8">
          <h1 className="text-3xl font-bold tracking-tight">EvoLens</h1>
          <p className="text-slate-300 mt-1 text-base">
            Evaluation instrument for the calibration pillar of AI model validation
          </p>
          <div className="flex gap-4 mt-4 text-xs">
            <span className="bg-white/10 px-3 py-1 rounded-full">BLEU</span>
            <span className="bg-white/10 px-3 py-1 rounded-full">ROUGE</span>
            <span className="bg-white/10 px-3 py-1 rounded-full">BERTScore</span>
            <span className="bg-white/10 px-3 py-1 rounded-full">Perplexity</span>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-8 space-y-8">
        {/* Three-pillar overview */}
        <section className="bg-white border border-slate-200 rounded-lg p-6">
          <h2 className="text-lg font-semibold text-navy mb-3">Three-Pillar Validation Framework</h2>
          <p className="text-sm text-slate-600 mb-4">
            Credit risk model validation under SR 11-7 rests on three pillars. LLM evaluation is
            rebuilding the same framework. EvoLens v1 implements the Calibration pillar.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="p-3 rounded-lg bg-navy/5 border border-navy/10">
              <div className="text-xs font-bold text-navy uppercase tracking-wider mb-1">
                Discrimination
              </div>
              <p className="text-xs text-slate-600">
                Does the model separate good from bad? LLM-as-judge, Elo, pairwise preference.
              </p>
              <span className="text-xs text-slate-400 mt-1 block italic">Planned for v2</span>
            </div>
            <div className="p-3 rounded-lg bg-navy/10 border-2 border-navy/20">
              <div className="text-xs font-bold text-navy uppercase tracking-wider mb-1">
                Calibration <span className="text-red">(this tool)</span>
              </div>
              <p className="text-xs text-slate-600">
                Does the candidate match the reference? BLEU, ROUGE, BERTScore, Perplexity.
              </p>
            </div>
            <div className="p-3 rounded-lg bg-navy/5 border border-navy/10">
              <div className="text-xs font-bold text-navy uppercase tracking-wider mb-1">
                Stability
              </div>
              <p className="text-xs text-slate-600">
                Is model behavior drifting? Perplexity drift, embedding drift.
              </p>
              <span className="text-xs text-slate-400 mt-1 block italic">Planned for v2</span>
            </div>
          </div>
        </section>

        {/* Mode toggle */}
        <div className="flex gap-3">
          <button
            onClick={() => { setShowBYO(false); setByoScores(null); }}
            className={`px-4 py-2 rounded-lg text-sm font-semibold cursor-pointer transition-colors ${
              !showBYO ? 'bg-navy text-white' : 'bg-white text-slate-600 border border-slate-200 hover:bg-slate-50'
            }`}
          >
            Pre-computed examples
          </button>
          <button
            onClick={() => setShowBYO(true)}
            className={`px-4 py-2 rounded-lg text-sm font-semibold cursor-pointer transition-colors ${
              showBYO ? 'bg-navy text-white' : 'bg-white text-slate-600 border border-slate-200 hover:bg-slate-50'
            }`}
          >
            Evaluate your own
          </button>
        </div>

        {/* Example selector or BYO panel */}
        {!showBYO ? (
          <section>
            <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-3">
              Select an example scenario
            </h2>
            <ExampleSelector examples={examples} selected={selected} onSelect={setSelected} />
          </section>
        ) : (
          <BYOPanel onResults={handleBYOResults} />
        )}

        {/* Text display */}
        {(isPrecomputed || byoScores) && (
          <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <TextDisplay label="Reference" text={refText} variant="reference" />
            <TextDisplay label="Candidate" text={candText} variant="candidate" />
          </section>
        )}

        {/* Metric cards */}
        {scores && (
          <section>
            <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-3">
              Metric scores
              {!isPrecomputed && (
                <span className="text-xs font-normal text-slate-400 ml-2">
                  (BLEU and ROUGE computed in-browser)
                </span>
              )}
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* BLEU */}
              {scores.bleu && (
                <MetricCard
                  metricId="bleu"
                  label="BLEU-4"
                  value={scores.bleu.bleu}
                  pedagogy={bleuPedagogy}
                  subScores={[
                    { label: 'BP', value: scores.bleu.brevity_penalty },
                    { label: 'P1', value: scores.bleu.p1 },
                    { label: 'P2', value: scores.bleu.p2 },
                    { label: 'P3', value: scores.bleu.p3 },
                    { label: 'P4', value: scores.bleu.p4 },
                  ]}
                />
              )}

              {/* ROUGE */}
              {scores.rouge1 && (
                <MetricCard
                  metricId="rouge"
                  label="ROUGE"
                  value={scores.rouge1.f1}
                  pedagogy={rougePedagogy}
                  subScores={[
                    { label: 'R1-F1', value: scores.rouge1.f1 },
                    { label: 'R2-F1', value: scores.rouge2?.f1 ?? 0 },
                    { label: 'RL-F1', value: scores.rougeL?.f1 ?? 0 },
                    { label: 'R1-R', value: scores.rouge1.recall },
                    { label: 'R1-P', value: scores.rouge1.precision },
                  ]}
                />
              )}

              {/* BERTScore */}
              {scores.bertscore && scores.bertscore.f1 > 0 && (
                <MetricCard
                  metricId="bertscore"
                  label="BERTScore"
                  value={scores.bertscore.f1_rescaled}
                  pedagogy={bertscorePedagogy}
                  subScores={[
                    { label: 'F1 (raw)', value: scores.bertscore.f1 },
                    { label: 'P (rescaled)', value: scores.bertscore.precision_rescaled },
                    { label: 'R (rescaled)', value: scores.bertscore.recall_rescaled },
                  ]}
                />
              )}

              {/* Perplexity */}
              {scores.perplexity && scores.perplexity.perplexity > 0 && (
                <MetricCard
                  metricId="perplexity"
                  label="Perplexity"
                  value={scores.perplexity.perplexity}
                  format="perplexity"
                  pedagogy={perplexityPedagogy}
                  subScores={[
                    { label: 'Avg log-prob', value: scores.perplexity.avg_log_prob },
                    { label: 'Tokens', value: scores.perplexity.num_tokens },
                  ]}
                />
              )}

              {/* Note when BERTScore/Perplexity unavailable in BYO mode */}
              {!isPrecomputed && (
                <div className="md:col-span-2 text-xs text-slate-400 italic p-4 bg-slate-50 border border-slate-200 rounded-lg">
                  BERTScore and Perplexity require ML models (BERT, GPT-2) and are available only
                  with pre-computed examples. The BLEU and ROUGE scores above were computed in your
                  browser using the same algorithms as the Python reference implementations.
                </div>
              )}
            </div>
          </section>
        )}

        {/* Description for selected example */}
        {isPrecomputed && (
          <section className="bg-white border border-slate-200 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-navy mb-1">Why this example matters</h3>
            <p className="text-sm text-slate-600">{selected.description}</p>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 mt-12">
        <div className="max-w-5xl mx-auto px-6 py-6 text-center">
          <p className="text-xs text-slate-400">
            Built by Paulo Cavallo, PhD. AI governance practitioner and author of{' '}
            <span className="font-semibold">AI Under Audit</span>.
          </p>
          <p className="text-xs text-slate-300 mt-1">
            All example data is synthetic. No real financial, client, or proprietary data is used.
          </p>
        </div>
      </footer>
    </div>
  );
}
