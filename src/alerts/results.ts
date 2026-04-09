// NCAAM Oracle v4.1 — Results Scorer
// Fetches yesterday's final scores, marks predictions correct/incorrect,
// updates season accuracy tracker.

import { logger } from '../logger.js';
import { fetchSchedule } from '../api/cbbClient.js';
import {
  getUnscoredPredictions,
  markPredictionResult,
  getSeasonRecord,
  upsertSeasonAccuracy,
  persistDb,
} from '../db/database.js';
import { getCurrentSeason } from '../season-manager/index.js';
import type { Prediction } from '../types.js';

// ─── Score yesterday's games ──────────────────────────────────────────────────

export async function scoreYesterdayGames(
  date: string
): Promise<{
  scored: number;
  correct: number;
  brierScore: number;
  accuracy: number;
}> {
  logger.info({ date }, 'Scoring previous day games');

  // Fetch games for that date (should include final scores if games are done)
  const games = await fetchSchedule(date);
  const finalGames = games.filter(g => g.status === 'Final');

  if (finalGames.length === 0) {
    logger.warn({ date }, 'No final games found for scoring');
    return { scored: 0, correct: 0, brierScore: 0, accuracy: 0 };
  }

  const unscored = getUnscoredPredictions(date);
  let scored = 0;
  let correct = 0;
  let brierSum = 0;

  for (const game of finalGames) {
    if (game.homeTeam.score === undefined || game.awayTeam.score === undefined) continue;

    const match = unscored.find(
      u => u.home_team === game.homeTeam.teamAbbr && u.away_team === game.awayTeam.teamAbbr
    );

    if (!match) {
      logger.debug({ gameId: game.gameId }, 'No unscored prediction found for game');
      continue;
    }

    markPredictionResult(
      game.gameId,
      date,
      game.homeTeam.score,
      game.awayTeam.score,
      game.homeTeam.teamAbbr,
      game.awayTeam.teamAbbr
    );

    const homeWon = game.homeTeam.score > game.awayTeam.score;
    const pickedHome = match.calibrated_prob >= 0.5;
    const isCorrect = pickedHome === homeWon;

    scored++;
    if (isCorrect) correct++;

    // Brier score: (pred - outcome)^2
    const outcome = homeWon ? 1 : 0;
    brierSum += Math.pow(match.calibrated_prob - outcome, 2);

    logger.debug(
      {
        game: `${game.awayTeam.teamAbbr} @ ${game.homeTeam.teamAbbr}`,
        score: `${game.awayTeam.score}-${game.homeTeam.score}`,
        correct: isCorrect,
      },
      'Game scored'
    );
  }

  const brierScore = scored > 0 ? brierSum / scored : 0;
  const accuracy   = scored > 0 ? correct / scored : 0;

  // Update season running totals
  if (scored > 0) {
    const season = getCurrentSeason();
    const seasonRecord = getSeasonRecord(season);

    upsertSeasonAccuracy({
      season,
      date,
      total_correct: seasonRecord.correct,
      total_picks: seasonRecord.total,
      accuracy_pct: seasonRecord.total > 0 ? seasonRecord.correct / seasonRecord.total : 0,
      high_conv_correct: seasonRecord.highConvCorrect,
      high_conv_total: seasonRecord.highConvTotal,
      brier: seasonRecord.brier,
      ats_wins: seasonRecord.atsWins,
      ats_losses: seasonRecord.atsLosses,
      tournament_correct: 0,
      tournament_total: 0,
    });

    persistDb();
  }

  logger.info({ date, scored, correct, brierScore: brierScore.toFixed(4), accuracy: (accuracy * 100).toFixed(1) + '%' }, 'Scoring complete');

  return { scored, correct, brierScore, accuracy };
}

// ─── Get yesterday's date ─────────────────────────────────────────────────────

export function getYesterdayDate(): string {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - 1);
  return d.toISOString().split('T')[0];
}
