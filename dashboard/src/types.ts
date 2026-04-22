export interface BleuScores {
  bleu: number;
  brevity_penalty: number;
  p1: number;
  p2: number;
  p3: number;
  p4: number;
}

export interface RougeVariantScores {
  precision: number;
  recall: number;
  f1: number;
}

export interface BertScoreScores {
  precision: number;
  recall: number;
  f1: number;
  precision_rescaled: number;
  recall_rescaled: number;
  f1_rescaled: number;
}

export interface PerplexityScores {
  perplexity: number;
  avg_log_prob: number;
  num_tokens: number;
}

export interface ExampleScores {
  bleu: BleuScores;
  rouge1: RougeVariantScores;
  rouge2: RougeVariantScores;
  rougeL: RougeVariantScores;
  bertscore: BertScoreScores;
  perplexity: PerplexityScores;
}

export interface Example {
  id: number;
  name: string;
  description: string;
  scenario: string;
  reference: string;
  candidate: string;
  scores: ExampleScores;
}

export interface ValueRange {
  range: string;
  interpretation: string;
}

export interface GoodBadValues {
  description: string;
  ranges: ValueRange[];
  caveat: string;
}

export interface PedagogyContent {
  id: string;
  name: string;
  fullName: string;
  pillar: string;
  pillarNote?: string;
  whatItMeasures: string;
  whyItMeasures: string;
  regulatorQuestions: string[];
  goodBadValues: GoodBadValues;
  learnMore: string;
  creditRiskParallel: string;
}
