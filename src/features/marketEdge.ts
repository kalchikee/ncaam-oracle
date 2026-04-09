// NCAAM Oracle v4.1 — Market Edge Detection
// CBB lines are less efficient than NBA/NFL.
// Best edges: mid-major games, early season before KenPom stabilizes, conf tourney upsets.

import type { ConfidenceTier, EdgeResult } from '../types.js';

// ─── Confidence tier classification ──────────────────────────────────────────

export function getConfidenceTier(calibratedProb: number): ConfidenceTier {
  const p = Math.max(calibratedProb, 1 - calibratedProb);
  if (p >= 0.75) return 'extreme';
  if (p >= 0.70) return 'high_conviction';
  if (p >= 0.63) return 'strong';
  if (p >= 0.55) return 'lean';
  return 'coin_flip';
}

export function confidenceEmoji(tier: ConfidenceTier): string {
  const map: Record<ConfidenceTier, string> = {
    extreme: '🚀',
    high_conviction: '🟢',
    strong: '✅',
    lean: '📊',
    coin_flip: '🪙',
  };
  return map[tier];
}

// ─── Edge detection ───────────────────────────────────────────────────────────

export function computeEdge(modelProb: number, vegasProb: number): EdgeResult {
  const edge = modelProb - vegasProb;
  const absEdge = Math.abs(edge);

  let edgeCategory: EdgeResult['edgeCategory'];
  if (absEdge < 0.03)      edgeCategory = 'none';
  else if (absEdge < 0.06) edgeCategory = 'small';
  else if (absEdge < 0.10) edgeCategory = 'meaningful';
  else                      edgeCategory = 'large';

  return { modelProb, vegasProb, edge, edgeCategory };
}

export function formatEdge(result: EdgeResult): string {
  const sign = result.edge >= 0 ? '+' : '';
  return (
    `Model: ${(result.modelProb * 100).toFixed(1)}%  ` +
    `Vegas: ${(result.vegasProb * 100).toFixed(1)}%  ` +
    `Edge: ${sign}${(result.edge * 100).toFixed(1)}%  [${result.edgeCategory.toUpperCase()}]`
  );
}

export function shouldHighlight(prob: number, edge?: number): boolean {
  const tier = getConfidenceTier(prob);
  if (tier === 'extreme' || tier === 'high_conviction') return true;
  if (edge !== undefined && Math.abs(edge) >= 0.06 && tier === 'strong') return true;
  return false;
}
