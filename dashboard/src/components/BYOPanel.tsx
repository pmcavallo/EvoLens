import { useState } from 'react';
import { computeBleu, computeRouge } from '../lib/metrics';
import type { ExampleScores } from '../types';

interface BYOPanelProps {
  onResults: (scores: Partial<ExampleScores>, reference: string, candidate: string) => void;
}

export default function BYOPanel({ onResults }: BYOPanelProps) {
  const [reference, setReference] = useState('');
  const [candidate, setCandidate] = useState('');
  const [prompt, setPrompt] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [provider, setProvider] = useState<'openai' | 'anthropic'>('openai');
  const [mode, setMode] = useState<'paste' | 'generate'>('paste');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function generateCandidate(): Promise<string> {
    if (provider === 'openai') {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: 'gpt-4o-mini',
          messages: [{ role: 'user', content: prompt }],
          max_tokens: 512,
        }),
      });
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error?.message || `OpenAI API error: ${response.status}`);
      }
      const data = await response.json();
      return data.choices[0].message.content;
    } else {
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01',
          'anthropic-dangerous-direct-browser-access': 'true',
        },
        body: JSON.stringify({
          model: 'claude-haiku-4-5-20251001',
          max_tokens: 512,
          messages: [{ role: 'user', content: prompt }],
        }),
      });
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error?.message || `Anthropic API error: ${response.status}`);
      }
      const data = await response.json();
      return data.content[0].text;
    }
  }

  async function handleEvaluate() {
    setError('');
    setLoading(true);

    try {
      let candidateText = candidate;

      if (mode === 'generate') {
        if (!apiKey || !prompt || !reference) {
          setError('Please provide a reference text, prompt, and API key.');
          setLoading(false);
          return;
        }
        candidateText = await generateCandidate();
        setCandidate(candidateText);
      } else {
        if (!reference || !candidateText) {
          setError('Please provide both reference and candidate text.');
          setLoading(false);
          return;
        }
      }

      // Compute BLEU and ROUGE in-browser
      const bleu = computeBleu(candidateText, reference);
      const rouge = computeRouge(candidateText, reference);

      onResults(
        { bleu, rouge1: rouge.rouge1, rouge2: rouge.rouge2, rougeL: rouge.rougeL },
        reference,
        candidateText
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-navy">Evaluate Your Own LLM Output</h3>
        <div className="flex gap-2">
          <button
            onClick={() => setMode('paste')}
            className={`text-xs px-3 py-1 rounded-full cursor-pointer ${
              mode === 'paste' ? 'bg-navy text-white' : 'bg-slate-100 text-slate-600'
            }`}
          >
            Paste text
          </button>
          <button
            onClick={() => setMode('generate')}
            className={`text-xs px-3 py-1 rounded-full cursor-pointer ${
              mode === 'generate' ? 'bg-navy text-white' : 'bg-slate-100 text-slate-600'
            }`}
          >
            Generate with API
          </button>
        </div>
      </div>

      <div className="space-y-3">
        <div>
          <label className="block text-xs font-semibold text-slate-600 mb-1">Reference text</label>
          <textarea
            value={reference}
            onChange={(e) => setReference(e.target.value)}
            placeholder="Enter the ground truth / reference text..."
            className="w-full border border-slate-200 rounded-lg p-3 text-sm text-slate-700 resize-y min-h-[80px] focus:outline-none focus:border-navy"
          />
        </div>

        {mode === 'paste' ? (
          <div>
            <label className="block text-xs font-semibold text-slate-600 mb-1">Candidate text</label>
            <textarea
              value={candidate}
              onChange={(e) => setCandidate(e.target.value)}
              placeholder="Enter the LLM output to evaluate..."
              className="w-full border border-slate-200 rounded-lg p-3 text-sm text-slate-700 resize-y min-h-[80px] focus:outline-none focus:border-navy"
            />
          </div>
        ) : (
          <>
            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1">Prompt</label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter the prompt to send to the LLM..."
                className="w-full border border-slate-200 rounded-lg p-3 text-sm text-slate-700 resize-y min-h-[80px] focus:outline-none focus:border-navy"
              />
            </div>

            <div className="flex gap-3">
              <div className="flex-1">
                <label className="block text-xs font-semibold text-slate-600 mb-1">Provider</label>
                <select
                  value={provider}
                  onChange={(e) => setProvider(e.target.value as 'openai' | 'anthropic')}
                  className="w-full border border-slate-200 rounded-lg p-2 text-sm focus:outline-none focus:border-navy"
                >
                  <option value="openai">OpenAI</option>
                  <option value="anthropic">Anthropic</option>
                </select>
              </div>
              <div className="flex-1">
                <label className="block text-xs font-semibold text-slate-600 mb-1">API Key</label>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder={provider === 'openai' ? 'sk-...' : 'sk-ant-...'}
                  className="w-full border border-slate-200 rounded-lg p-2 text-sm focus:outline-none focus:border-navy"
                />
              </div>
            </div>

            <p className="text-xs text-slate-400">
              Your API key is stored in-memory only. It is never persisted, logged, or sent anywhere
              other than the provider you selected.
            </p>
          </>
        )}

        {error && <p className="text-sm text-red">{error}</p>}

        <button
          onClick={handleEvaluate}
          disabled={loading}
          className="bg-navy text-white px-6 py-2 rounded-lg text-sm font-semibold hover:bg-navy-light transition-colors disabled:opacity-50 cursor-pointer"
        >
          {loading ? 'Evaluating...' : 'Evaluate'}
        </button>

        <p className="text-xs text-slate-400">
          BLEU and ROUGE are computed in your browser. BERTScore and Perplexity require ML models
          and are available only with the pre-computed examples.
        </p>
      </div>
    </div>
  );
}
