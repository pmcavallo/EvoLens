import { useState } from 'react';
import type { PedagogyContent } from '../types';

interface PedagogyPanelProps {
  content: PedagogyContent;
}

export default function PedagogyPanel({ content }: PedagogyPanelProps) {
  const [showLearnMore, setShowLearnMore] = useState(false);

  return (
    <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 mt-2 text-left animate-in">
      {/* What this number measures */}
      <div className="mb-3">
        <h5 className="text-xs font-semibold text-navy uppercase tracking-wider mb-1">
          What this number measures
        </h5>
        <p className="text-sm text-slate-700">{content.whatItMeasures}</p>
      </div>

      {/* Why it measures that */}
      <div className="mb-3">
        <h5 className="text-xs font-semibold text-navy uppercase tracking-wider mb-1">
          Why it measures that
        </h5>
        <p className="text-sm text-slate-600">{content.whyItMeasures}</p>
      </div>

      {/* What a regulator would ask */}
      <div className="mb-3">
        <h5 className="text-xs font-semibold text-red uppercase tracking-wider mb-1">
          What a regulator or MRM reviewer would ask
        </h5>
        <ul className="list-disc list-outside ml-4 space-y-1">
          {content.regulatorQuestions.map((q, i) => (
            <li key={i} className="text-sm text-slate-600">{q}</li>
          ))}
        </ul>
      </div>

      {/* Good/bad values */}
      <div className="mb-3">
        <h5 className="text-xs font-semibold text-navy uppercase tracking-wider mb-1">
          Interpretation guide
        </h5>
        <p className="text-xs text-slate-500 mb-2">{content.goodBadValues.description}</p>
        <div className="space-y-1">
          {content.goodBadValues.ranges.map((r, i) => (
            <div key={i} className="flex text-sm">
              <span className="font-mono text-navy font-semibold min-w-[140px] shrink-0">
                {r.range}
              </span>
              <span className="text-slate-600">{r.interpretation}</span>
            </div>
          ))}
        </div>
        <p className="text-xs text-slate-400 mt-2 italic">{content.goodBadValues.caveat}</p>
      </div>

      {/* Credit risk parallel */}
      <div className="mb-3 bg-navy/5 rounded p-3">
        <h5 className="text-xs font-semibold text-navy uppercase tracking-wider mb-1">
          Credit risk parallel
        </h5>
        <p className="text-sm text-slate-600 font-serif italic">{content.creditRiskParallel}</p>
      </div>

      {/* Learn more */}
      <button
        onClick={() => setShowLearnMore(!showLearnMore)}
        className="text-xs text-navy hover:text-navy-light font-semibold cursor-pointer"
      >
        {showLearnMore ? 'Hide details' : 'Learn more \u2192'}
      </button>

      {showLearnMore && (
        <div className="mt-2 p-3 bg-white border border-slate-100 rounded text-sm text-slate-600 whitespace-pre-line">
          {content.learnMore}
        </div>
      )}
    </div>
  );
}
