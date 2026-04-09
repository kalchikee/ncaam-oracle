// NCAAM Oracle v4.1 — Daily Prediction Pipeline
// Orchestrates: Season gate → Score yesterday → Fetch schedule → Feature engineering
//   → Monte Carlo → ML model → Edge detection → Store → Discord embed

import { logger } from './logger.js';
import { fetchSchedule, fetchAllTeamStats } from './api/cbbClient.js';
import { initializeOdds, hasAnyOdds, getOddsForGame } from './api/oddsClient.js';
import { computeFeatures, earlySeasonLabel } from './features/featureEngine.js';
import { computeEdge } from './features/marketEdge.js';
import { runMonteCarlo } from './models/monteCarlo.js';
import { loadModel, predict as mlPredict, isModelLoaded, getModelInfo } from './models/metaModel.js';
import { upsertPrediction, initDb, persistDb } from './db/database.js';
import { getSeasonPhase, getCurrentSeason, isInSeason } from './season-manager/index.js';
import type { CBBGame, Prediction, PipelineOptions } from './types.js';

const MODEL_VERSION = '4.1.0';

// ─── Main pipeline ────────────────────────────────────────────────────────────

export async function runPipeline(options: PipelineOptions = {}): Promise<Prediction[]> {
  const today = new Date().toISOString().split('T')[0];
  const gameDate = options.date ?? today;

  logger.info({ gameDate, version: MODEL_VERSION }, '=== NCAAM Oracle v4.1 Pipeline Start ===');

  // 1. Season gate
  if (!options.tournamentMode && !isInSeason(gameDate)) {
    const phase = getSeasonPhase(gameDate);
    logger.info({ phase, date: gameDate }, 'Outside active season window — exiting');
    return [];
  }

  const phase = getSeasonPhase(gameDate);
  logger.info({ phase, date: gameDate }, 'Season gate passed');

  // 2. Initialize database
  await initDb();

  // 3. Load ML meta-model
  const modelLoaded = loadModel();
  if (modelLoaded) {
    const info = getModelInfo();
    logger.info(
      { version: info?.version, avgBrier: info?.avgBrier, seasons: info?.trainSeasons },
      'ML meta-model active'
    );
  } else {
    logger.info('ML model not found — Monte Carlo fallback active');
    logger.info('Run: npm run train  to train the model (Python required)');
  }

  // 4. Initialize Vegas odds
  await initializeOdds(gameDate);
  if (hasAnyOdds()) {
    logger.info('Vegas lines loaded — edge detection active');
  }

  // (stats are pre-fetched per game below)

  // 6. Fetch today's schedule
  const games = await fetchSchedule(gameDate);
  const scheduledGames = games.filter(g => g.status === 'Scheduled' || g.status === 'Live');

  if (scheduledGames.length === 0) {
    logger.warn({ gameDate }, 'No upcoming games found');
    return [];
  }

  logger.info({ gameDate, scheduled: scheduledGames.length, total: games.length }, 'Schedule fetched');

  // 6b. Pre-fetch team stats for all teams playing today
  await fetchAllTeamStats(scheduledGames);

  // 7. Process each game
  const predictions: Prediction[] = [];

  for (const game of scheduledGames) {
    try {
      const pred = await processGame(game, gameDate, modelLoaded, phase === 'march_madness');
      if (pred) predictions.push(pred);
    } catch (err) {
      logger.error(
        { err, home: game.homeTeam.teamAbbr, away: game.awayTeam.teamAbbr },
        'Failed to process game'
      );
    }
  }

  persistDb();

  logger.info(
    { processed: predictions.length, total: scheduledGames.length },
    '=== Pipeline Complete ==='
  );

  if (options.verbose !== false) {
    printPredictions(predictions, gameDate, modelLoaded);
  }

  return predictions;
}

// ─── Single game processing ───────────────────────────────────────────────────

async function processGame(
  game: CBBGame,
  gameDate: string,
  modelLoaded: boolean,
  isTournament: boolean
): Promise<Prediction | null> {
  const homeAbbr = game.homeTeam.teamAbbr;
  const awayAbbr = game.awayTeam.teamAbbr;

  logger.debug({ matchup: `${awayAbbr} @ ${homeAbbr}` }, 'Processing game');

  // Step A: Feature engineering (with prior-year bootstrap)
  const features = await computeFeatures(game, gameDate);

  // Step B: Monte Carlo simulation
  const mc = runMonteCarlo(features);

  // Step C: ML calibration
  let calibrated_prob: number;

  if (modelLoaded && isModelLoaded()) {
    calibrated_prob = mlPredict(features, mc.win_probability);
  } else {
    calibrated_prob = mc.win_probability;
  }

  // Step D: Market edge
  let vegas_prob: number | undefined;
  let edge: number | undefined;

  const matchupKey = `${awayAbbr}@${homeAbbr}`;
  const odds = getOddsForGame(matchupKey);

  if (odds) {
    vegas_prob = odds.homeImpliedProb;
    edge = calibrated_prob - vegas_prob;
    features.vegas_home_prob = vegas_prob;
    const edgeResult = computeEdge(calibrated_prob, vegas_prob);
    logger.info({ matchup: matchupKey, edge: edgeResult.edgeCategory }, 'Edge detected');
  }

  // Step E: Early season label
  const minGP = Math.min(features.home_games_played, features.away_games_played);
  const earlyLabel = earlySeasonLabel(minGP);

  // Step F: Build prediction
  const prediction: Prediction = {
    game_date: gameDate,
    game_id: game.gameId,
    home_team: homeAbbr,
    away_team: awayAbbr,
    arena: game.arena,
    is_neutral_site: game.isNeutralSite,
    is_tournament: game.isTournamentGame || isTournament,
    feature_vector: features,
    mc_win_pct: mc.win_probability,
    calibrated_prob,
    vegas_prob,
    edge,
    model_version: MODEL_VERSION,
    home_exp_pts: mc.home_exp_pts,
    away_exp_pts: mc.away_exp_pts,
    total_points: mc.total_points,
    spread: mc.spread,
    most_likely_score: `${mc.most_likely_score[0]}-${mc.most_likely_score[1]}`,
    early_season_label: earlyLabel,
    created_at: new Date().toISOString(),
  };

  // Step G: Store
  upsertPrediction(prediction);

  return prediction;
}

// ─── Console output ───────────────────────────────────────────────────────────

function printPredictions(
  predictions: Prediction[],
  gameDate: string,
  mlActive: boolean
): void {
  if (predictions.length === 0) {
    console.log(`\nNo predictions for ${gameDate}\n`);
    return;
  }

  const modelLabel = mlActive ? 'ML+Isotonic' : 'Monte Carlo';
  const width = 110;

  console.log('\n' + '═'.repeat(width));
  console.log(
    `  NCAAM ORACLE v4.1  ·  ${gameDate}  ·  ${predictions.length} games  ·  [${modelLabel}]`
  );
  console.log('═'.repeat(width));
  console.log(
    `\n  ${pad('MATCHUP', 26)}  ${pad('CAL WIN%', 10)}  ${pad('MC WIN%', 8)}  ${pad('SPREAD', 18)}  ${pad('O/U', 6)}  ${pad('SCORE', 9)}  EDGE`
  );
  console.log('─'.repeat(width));

  const sorted = [...predictions].sort(
    (a, b) => Math.abs(b.calibrated_prob - 0.5) - Math.abs(a.calibrated_prob - 0.5)
  );

  for (const p of sorted) {
    const calPct = (p.calibrated_prob * 100).toFixed(1) + '%';
    const mcPct  = (p.mc_win_pct * 100).toFixed(1) + '%';
    const conf   = Math.abs(p.calibrated_prob - 0.5);
    const marker = conf >= 0.25 ? '🚀' : conf >= 0.20 ? '🟢' : conf >= 0.13 ? '✅' : '  ';

    const spreadStr = p.spread >= 0
      ? `${p.home_team} -${Math.abs(p.spread).toFixed(1)}`
      : `${p.away_team} -${Math.abs(p.spread).toFixed(1)}`;

    const edgeStr = p.edge !== undefined
      ? (p.edge >= 0 ? '+' : '') + (p.edge * 100).toFixed(1) + '%'
      : '—';

    const early = p.early_season_label ? ' *' : '';

    console.log(
      `${marker} ${pad(`${p.away_team} @ ${p.home_team}`, 26)}  ${pad(calPct, 10)}  ${pad(mcPct, 8)}  ${pad(spreadStr, 18)}  ${pad(p.total_points.toFixed(0), 6)}  ${pad(p.most_likely_score, 9)}  ${edgeStr}${early}`
    );
  }

  console.log('─'.repeat(width));
  console.log('\n  🚀 = Extreme (75%+)  🟢 = High Conviction (70–75%)  ✅ = Strong (63–70%)  * = Early Season');

  const highConv = predictions.filter(p => Math.abs(p.calibrated_prob - 0.5) >= 0.20).length;
  const edgePicks = predictions.filter(p => p.edge !== undefined && Math.abs(p.edge) >= 0.06).length;

  console.log(
    `\n  Summary: ${predictions.length} games · ${highConv} high-conviction picks · ${edgePicks} edge picks (≥6%) · ` +
    `Avg O/U: ${(predictions.reduce((s, p) => s + p.total_points, 0) / predictions.length).toFixed(0)}\n`
  );
  console.log('═'.repeat(width) + '\n');
}

function pad(str: string, width: number): string {
  if (str.length >= width) return str.slice(0, width);
  return str + ' '.repeat(width - str.length);
}
