import { useState } from 'react';

interface PillarBadgeProps {
  pillar: string;
  note?: string;
}

const PILLAR_INFO = {
  Calibration: {
    color: 'bg-navy text-white',
    description:
      'Does the candidate match the reference? The Calibration pillar asks whether predicted outputs match observed or expected outputs. In credit risk terms: does the predicted PD match the realized default rate?',
  },
  Discrimination: {
    color: 'bg-slate-600 text-white',
    description:
      'Can the system rank outputs by quality? The Discrimination pillar asks whether the evaluation system can distinguish good outputs from bad. In credit risk terms: Somers\u2019 D, KS, Gini, AUC.',
  },
  Stability: {
    color: 'bg-slate-500 text-white',
    description:
      'Is model behavior drifting? The Stability pillar asks whether the output distribution is shifting over time. In credit risk terms: PSI (Population Stability Index).',
  },
};

export default function PillarBadge({ pillar, note }: PillarBadgeProps) {
  const [expanded, setExpanded] = useState(false);
  const info = PILLAR_INFO[pillar as keyof typeof PILLAR_INFO] ?? PILLAR_INFO.Calibration;

  return (
    <div className="inline-block relative">
      <button
        onClick={() => setExpanded(!expanded)}
        className={`${info.color} text-xs font-semibold px-2.5 py-0.5 rounded-full cursor-pointer hover:opacity-90 transition-opacity`}
      >
        {pillar}
      </button>

      {expanded && (
        <div className="absolute z-20 top-8 left-0 w-80 bg-white border border-slate-200 rounded-lg shadow-lg p-4 text-left">
          <h4 className="text-sm font-bold text-navy mb-2">Three-Pillar Validation Framework</h4>
          <p className="text-xs text-slate-600 mb-3">
            Credit risk model validation under SR 11-7 rests on three pillars. LLM evaluation
            is rebuilding this same framework.
          </p>

          <div className="space-y-2">
            {Object.entries(PILLAR_INFO).map(([name, p]) => (
              <div
                key={name}
                className={`p-2 rounded text-xs ${name === pillar ? 'bg-slate-50 border border-navy/20' : 'bg-slate-50/50'}`}
              >
                <span className={`${p.color} text-xs px-1.5 py-0.5 rounded-full mr-1.5`}>
                  {name}
                </span>
                <span className="text-slate-600">{p.description}</span>
              </div>
            ))}
          </div>

          {note && (
            <p className="text-xs text-slate-500 mt-3 pt-2 border-t border-slate-100 italic">
              {note}
            </p>
          )}

          <p className="text-xs text-slate-400 mt-3 pt-2 border-t border-slate-100">
            EvoLens v1 covers the Calibration pillar. Discrimination and Stability are planned
            extensions.
          </p>
        </div>
      )}
    </div>
  );
}
