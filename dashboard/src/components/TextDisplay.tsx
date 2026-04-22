interface TextDisplayProps {
  label: string;
  text: string;
  variant: 'reference' | 'candidate';
}

export default function TextDisplay({ label, text, variant }: TextDisplayProps) {
  const borderColor = variant === 'reference' ? 'border-l-navy' : 'border-l-red';
  const labelColor = variant === 'reference' ? 'text-navy' : 'text-red';

  return (
    <div className={`border-l-3 ${borderColor} pl-4 py-2`}>
      <div className={`text-xs font-semibold uppercase tracking-wider ${labelColor} mb-1`}>
        {label}
      </div>
      <p className="text-sm text-slate-700 leading-relaxed">{text}</p>
    </div>
  );
}
