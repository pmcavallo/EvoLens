/**
 * In-browser implementations of BLEU and ROUGE for the BYO mode.
 * BERTScore and Perplexity require ML models and use pre-computed values.
 */

function tokenize(text: string): string[] {
  return text.toLowerCase().split(/\s+/).filter(Boolean);
}

function countNgrams(tokens: string[], n: number): Map<string, number> {
  const counts = new Map<string, number>();
  for (let i = 0; i <= tokens.length - n; i++) {
    const ngram = tokens.slice(i, i + n).join(' ');
    counts.set(ngram, (counts.get(ngram) || 0) + 1);
  }
  return counts;
}

// --- BLEU ---

export function computeBleu(
  candidate: string,
  reference: string,
  maxN = 4,
  epsilon = 0.1
): {
  bleu: number;
  brevity_penalty: number;
  p1: number;
  p2: number;
  p3: number;
  p4: number;
} {
  const candTokens = tokenize(candidate);
  const refTokens = tokenize(reference);
  const c = candTokens.length;
  const r = refTokens.length;

  if (c === 0) {
    return { bleu: 0, brevity_penalty: 0, p1: 0, p2: 0, p3: 0, p4: 0 };
  }

  // Brevity penalty
  const bp = c > r ? 1.0 : Math.exp(1 - r / c);

  const precisions: number[] = [];

  for (let n = 1; n <= maxN; n++) {
    const candNgrams = countNgrams(candTokens, n);
    const refNgrams = countNgrams(refTokens, n);

    let matches = 0;
    let total = 0;
    for (const [ngram, count] of candNgrams) {
      matches += Math.min(count, refNgrams.get(ngram) || 0);
      total += count;
    }

    if (total === 0) {
      precisions.push(0);
      continue;
    }

    const adjustedMatches = matches === 0 ? epsilon : matches;
    precisions.push(adjustedMatches / total);
  }

  // Geometric mean in log space
  if (precisions.some((p) => p === 0)) {
    return {
      bleu: 0,
      brevity_penalty: bp,
      p1: precisions[0] || 0,
      p2: precisions[1] || 0,
      p3: precisions[2] || 0,
      p4: precisions[3] || 0,
    };
  }

  const logAvg = precisions.reduce((sum, p) => sum + Math.log(p), 0) / maxN;
  const bleu = bp * Math.exp(logAvg);

  return {
    bleu,
    brevity_penalty: bp,
    p1: precisions[0],
    p2: precisions[1],
    p3: precisions[2] || 0,
    p4: precisions[3] || 0,
  };
}

// --- ROUGE ---

function rougeN(candTokens: string[], refTokens: string[], n: number) {
  const candNgrams = countNgrams(candTokens, n);
  const refNgrams = countNgrams(refTokens, n);

  let matches = 0;
  for (const [ngram, count] of refNgrams) {
    matches += Math.min(count, candNgrams.get(ngram) || 0);
  }

  const refTotal = Math.max([...refNgrams.values()].reduce((a, b) => a + b, 0), 1);
  const candTotal = Math.max([...candNgrams.values()].reduce((a, b) => a + b, 0), 1);

  const recall = matches / refTotal;
  const precision = matches / candTotal;
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);

  return { precision, recall, f1 };
}

function lcsLength(a: string[], b: string[]): number {
  const m = a.length;
  const n = b.length;
  let prev = new Array(n + 1).fill(0);
  let curr = new Array(n + 1).fill(0);

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1] === b[j - 1]) {
        curr[j] = prev[j - 1] + 1;
      } else {
        curr[j] = Math.max(prev[j], curr[j - 1]);
      }
    }
    [prev, curr] = [curr, new Array(n + 1).fill(0)];
  }
  return prev[n];
}

function rougeL(candTokens: string[], refTokens: string[]) {
  const lcs = lcsLength(candTokens, refTokens);
  const refLen = Math.max(refTokens.length, 1);
  const candLen = Math.max(candTokens.length, 1);

  const recall = lcs / refLen;
  const precision = lcs / candLen;
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);

  return { precision, recall, f1 };
}

export function computeRouge(candidate: string, reference: string) {
  const candTokens = tokenize(candidate);
  const refTokens = tokenize(reference);

  return {
    rouge1: rougeN(candTokens, refTokens, 1),
    rouge2: rougeN(candTokens, refTokens, 2),
    rougeL: rougeL(candTokens, refTokens),
  };
}
