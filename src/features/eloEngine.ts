// NCAAM Oracle v4.1 — Elo Rating Engine
// CBB-specific tuning:
//   K-factor: 30 (higher variance than NBA/NFL due to game-to-game swings)
//   Margin cap: 25 pts (CBB blowouts are common, reduce their signal)
//   Offseason regression: 50% carry + 25% recruiting Elo + 25% D-I mean
//   Home advantage: +100 Elo points in expected score

import { getElo, upsertElo } from '../db/database.js';
import { logger } from '../logger.js';

export const LEAGUE_MEAN_ELO = 1500;
const K_FACTOR = 30;
const HOME_ADV_ELO = 100;
const MOV_CAP = 25;

// ─── Win probability from Elo ────────────────────────────────────────────────

export function eloWinProb(homeElo: number, awayElo: number, isNeutral = false): number {
  const homeAdj = isNeutral ? 0 : HOME_ADV_ELO;
  const diff = (homeElo + homeAdj) - awayElo;
  return 1 / (1 + Math.pow(10, -diff / 400));
}

export function getEloDiff(homeAbbr: string, awayAbbr: string): number {
  return getElo(homeAbbr) - getElo(awayAbbr);
}

export function getEloWinProb(homeAbbr: string, awayAbbr: string, isNeutral = false): number {
  return eloWinProb(getElo(homeAbbr), getElo(awayAbbr), isNeutral);
}

// ─── Elo update after game ───────────────────────────────────────────────────

export function updateEloAfterGame(
  homeAbbr: string,
  awayAbbr: string,
  homeScore: number,
  awayScore: number,
  isNeutral = false
): void {
  const homeElo = getElo(homeAbbr);
  const awayElo = getElo(awayAbbr);

  const homeExpected = eloWinProb(homeElo, awayElo, isNeutral);
  const homeActual = homeScore > awayScore ? 1 : 0;

  const margin = Math.abs(homeScore - awayScore);
  const movMult = Math.log(1 + Math.min(margin, MOV_CAP));
  const adjK = K_FACTOR * movMult;

  const homeNew = homeElo + adjK * (homeActual - homeExpected);
  const awayNew = awayElo + adjK * ((1 - homeActual) - (1 - homeExpected));

  const now = new Date().toISOString();
  upsertElo({ teamAbbr: homeAbbr, rating: Math.round(homeNew), updatedAt: now });
  upsertElo({ teamAbbr: awayAbbr, rating: Math.round(awayNew), updatedAt: now });

  logger.debug(
    { homeAbbr, awayAbbr, homeElo: homeNew.toFixed(0), awayElo: awayNew.toFixed(0) },
    'Elo updated'
  );
}

// ─── Offseason regression ────────────────────────────────────────────────────
// new_elo = 0.50 × final_elo + 0.25 × recruiting_elo + 0.25 × D1_mean
// Recruiting Elo: convert 247 composite rank to Elo (rank 1 = 1700, rank 364 = 1300)

export function applyOffseasonRegression(
  teamAbbr: string,
  recruitingRank: number // 1–364; lower is better
): void {
  const currentElo = getElo(teamAbbr);
  const recruitingElo = 1700 - (recruitingRank - 1) * (400 / 363);
  const newElo = 0.50 * currentElo + 0.25 * recruitingElo + 0.25 * LEAGUE_MEAN_ELO;

  upsertElo({
    teamAbbr,
    rating: Math.round(newElo),
    updatedAt: new Date().toISOString(),
  });
}

// ─── Seed teams that have no Elo yet ─────────────────────────────────────────

export function seedElo(teamAbbr: string, initialRating = LEAGUE_MEAN_ELO): void {
  const existing = getElo(teamAbbr);
  if (existing === LEAGUE_MEAN_ELO) {
    // Insert default if not already stored
    upsertElo({ teamAbbr, rating: initialRating, updatedAt: new Date().toISOString() });
  }
}
