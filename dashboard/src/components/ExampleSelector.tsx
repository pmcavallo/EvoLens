import type { Example } from '../types';

interface ExampleSelectorProps {
  examples: Example[];
  selected: Example;
  onSelect: (example: Example) => void;
}

const SCENARIO_ICONS: Record<string, string> = {
  paraphrase: '\uD83D\uDD04',
  reorder: '\u21C4',
  hallucination: '\u26A0\uFE0F',
  undercoverage: '\uD83D\uDD0D',
  unrelated: '\u2716',
  fluent_wrong: '\uD83C\uDFAD',
  gist_summary: '\uD83D\uDCDD',
  mixed_quality: '\u2696\uFE0F',
};

export default function ExampleSelector({ examples, selected, onSelect }: ExampleSelectorProps) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
      {examples.map((ex) => (
        <button
          key={ex.id}
          onClick={() => onSelect(ex)}
          className={`text-left p-3 rounded-lg border transition-all cursor-pointer ${
            selected.id === ex.id
              ? 'border-navy bg-navy/5 shadow-sm'
              : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50'
          }`}
        >
          <div className="flex items-start gap-1.5">
            <span className="text-base shrink-0">{SCENARIO_ICONS[ex.scenario] || '\uD83D\uDCCA'}</span>
            <div>
              <div className="text-xs font-semibold text-slate-800 leading-tight">{ex.name}</div>
              <div className="text-xs text-slate-400 mt-0.5 leading-tight line-clamp-2">{ex.description}</div>
            </div>
          </div>
        </button>
      ))}
    </div>
  );
}
