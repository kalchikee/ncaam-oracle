// NCAAM Oracle v4.1 — Monte Carlo Simulation Engine
// 10,000 Normal-distribution simulations
// Expected points:
//   exp_pts = team_AdjOE × (opp_AdjDE / D1_avg) × expected_possessions / 100 × home_adj × rest_adj
// CBB averages: ~68 possessions/40 min, ~72 pts/game

import type { FeatureVector, MonteCarloResult } from '../types.js';

const N_SIMULATIONS = 10_000;
const D1_AVG_OE    = 100.0;
const D1_AVG_DE    = 100.0;
const D1_AVG_TEMPO = 68.0;   // possessions per 40 min (D-I average)
const D1_AVG_PTS   = 72.5;
const CBB_SCORE_STD = 10.5;  // standard deviation of a CBB team's per-game score

// ─── Normal random (Box-Muller) ───────────────────────────────────────────────

function normalRandom(mean: number, std: number): number {
  const u1 = Math.random();
  const u2 = Math.random();
  const z = Math.sqrt(-2 * Math.log(Math.max(1e-10, u1))) * Math.cos(2 * Math.PI * u2);
  return mean + z * std;
}

// ─── Expected possessions from tempo interaction ───────────────────────────────
// expected_possessions = 0.45 × fast_team_tempo + 0.55 × slow_team_tempo

function expectedPossessions(homeTempo: number, awayTempo: number): number {
  const fast = Math.max(homeTempo, awayTempo);
  const slow = Math.min(homeTempo, awayTempo);
  return 0.45 * fast + 0.55 * slow;
}

// ─── Expected points ──────────────────────────────────────────────────────────

function estimateExpectedPoints(features: FeatureVector): {
  homeExpPts: number;
  awayExpPts: number;
  homeStd: number;
  awayStd: number;
  possessions: number;
} {
  const homeAdjOE = D1_AVG_OE + features.adj_oe_diff / 2;
  const awayAdjOE  = D1_AVG_OE - features.adj_oe_diff / 2;

  const homeAdjDE = D1_AVG_DE + features.adj_de_diff / 2;  // lower = better
  const awayAdjDE  = D1_AVG_DE - features.adj_de_diff / 2;

  const homeTempo = D1_AVG_TEMPO + features.adj_tempo_diff / 2;
  const awayTempo  = D1_AVG_TEMPO - features.adj_tempo_diff / 2;

  const poss = expectedPossessions(homeTempo, awayTempo);

  // Raw expected points per team = (AdjOE × opp_AdjDE / D1_avg × poss / 100)
  const homeRaw = homeAdjOE * (awayAdjDE / D1_AVG_DE) * (poss / 100);
  const awayRaw  = awayAdjOE  * (homeAdjDE / D1_AVG_DE) * (poss / 100);

  // Home court adjustment
  const hca = features.is_home ? features.home_court_advantage : 0;

  // Rest adjustment (each day of extra rest ≈ +0.4 pts for CBB)
  const restBonus = Math.max(-4, Math.min(4, features.rest_days_diff * 0.4));

  // B2B penalty: -2.5 pts for CBB
  const b2bPenalty = features.b2b_flag ? 2.5 : 0;

  const homeExpPts = Math.max(55, homeRaw + hca + restBonus - b2bPenalty / 2);
  const awayExpPts  = Math.max(55, awayRaw  - hca - restBonus - b2bPenalty / 2);

  // Higher tempo → higher variance
  const tempoFactor = poss / D1_AVG_TEMPO;
  const homeStd = CBB_SCORE_STD * tempoFactor;
  const awayStd  = CBB_SCORE_STD * tempoFactor;

  return { homeExpPts, awayExpPts, homeStd, awayStd, possessions: poss };
}

// ─── Overtime handling ────────────────────────────────────────────────────────
// OT: +8 points per team per OT period in CBB

function simulateOT(homeScore: number, awayScore: number, maxOT = 4): [number, number] {
  let h = homeScore;
  let a = awayScore;
  let otPeriods = 0;

  while (h === a && otPeriods < maxOT) {
    // Each OT period adds ~8 pts per team with variance
    h += Math.round(Math.max(0, normalRandom(8, 3)));
    a += Math.round(Math.max(0, normalRandom(8, 3)));
    otPeriods++;
  }

  // Force resolution if still tied after maxOT
  if (h === a) h += 1;
  return [h, a];
}

// ─── Main simulation ──────────────────────────────────────────────────────────

export function runMonteCarlo(features: FeatureVector): MonteCarloResult {
  const { homeExpPts, awayExpPts, homeStd, awayStd, possessions } = estimateExpectedPoints(features);

  let homeWins = 0;
  let totalHomeScore = 0;
  let totalAwayScore = 0;
  const homeScores: number[] = [];
  const awayScores: number[] = [];

  for (let i = 0; i < N_SIMULATIONS; i++) {
    let h = Math.round(normalRandom(homeExpPts, homeStd));
    let a = Math.round(normalRandom(awayExpPts, awayStd));
    h = Math.max(40, h);
    a = Math.max(40, a);

    if (h === a) {
      [h, a] = simulateOT(h, a);
    }

    if (h > a) homeWins++;
    totalHomeScore += h;
    totalAwayScore += a;
    homeScores.push(h);
    awayScores.push(a);
  }

  const winProb = homeWins / N_SIMULATIONS;
  const avgHome = totalHomeScore / N_SIMULATIONS;
  const avgAway  = totalAwayScore / N_SIMULATIONS;

  // Most likely score = rounded expected values
  const mostLikelyHome = Math.round(homeExpPts);
  const mostLikelyAway  = Math.round(awayExpPts);

  // Spread: positive = home team favored by that many points
  const spread = avgHome - avgAway;
  const total  = avgHome + avgAway;

  return {
    win_probability: Math.max(0.01, Math.min(0.99, winProb)),
    away_win_probability: Math.max(0.01, Math.min(0.99, 1 - winProb)),
    spread,
    total_points: total,
    most_likely_score: [mostLikelyHome, mostLikelyAway],
    home_exp_pts: avgHome,
    away_exp_pts: avgAway,
    simulations: N_SIMULATIONS,
  };
}
