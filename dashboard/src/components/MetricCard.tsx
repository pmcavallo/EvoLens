import { useState } from 'react';
import PillarBadge from './PillarBadge';
import PedagogyPanel from './PedagogyPanel';
import type { PedagogyContent } from '../types';

interface MetricCardProps {
  metricId: string;
  label: string;
  value: number;
  format?: 'score' | 'perplexity';
  subScores?: { label: string; value: number }[];
  pedagogy: PedagogyContent;
}

function formatScore(value: number, format: 'score' | 'perplexity'): string {
  if (format === 'perplexity') {
    return value.toFixed(1);
  }
  return value.toFixed(3);
}

function getScoreColor(value: number, format: 'score' | 'perplexity'): string {
  if (format === 'perplexity') {
    if (value < 15) return 'text-emerald-600';
    if (value < 30) return 'text-navy';
    if (value < 100) return 'text-amber-600';
    return 'text-red';
  }
  if (value >= 0.7) return 'text-emerald-600';
  if (value >= 0.4) return 'text-navy';
  if (value >= 0.2) return 'text-amber-600';
  return 'text-red';
}

export default function MetricCard({
  label,
  value,
  format = 'score',
  subScores,
  pedagogy,
}: MetricCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left p-4 hover:bg-slate-50/50 transition-colors cursor-pointer"
      >
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-slate-800">{label}</h3>
            <PillarBadge pillar={pedagogy.pillar} note={pedagogy.pillarNote} />
          </div>
          <span className="text-xs text-slate-400">{expanded ? '\u25B2' : 'Click to learn \u25BC'}</span>
        </div>

        <div className={`text-2xl font-bold font-mono ${getScoreColor(value, format)}`}>
          {formatScore(value, format)}
          {format === 'perplexity' && <span className="text-sm font-normal text-slate-400 ml-1">PPL</span>}
        </div>

        {subScores && subScores.length > 0 && (
          <div className="flex gap-3 mt-2">
            {subScores.map((s) => (
              <div key={s.label} className="text-xs">
                <span className="text-slate-400">{s.label}:</span>{' '}
                <span className="font-mono text-slate-600">{s.value.toFixed(3)}</span>
              </div>
            ))}
          </div>
        )}
      </button>

      {expanded && (
        <div className="px-4 pb-4">
          <PedagogyPanel content={pedagogy} />
        </div>
      )}
    </div>
  );
}
