// NCAAM Oracle v4.1 — CLI Entry Point
// Usage:
//   npm start                              → predictions for today
//   npm start -- --date 2025-11-15        → specific date
//   npm start -- --alert daily            → daily Discord embed
//   npm start -- --alert weekly           → weekly recap embed
//   npm start -- --alert bracket          → March Madness bracket embed
//   npm start -- --alert preseason        → preseason online embed
//   npm start -- --alert season-summary   → final season summary embed
//   npm start -- --score YYYY-MM-DD       → score yesterday's games

import 'dotenv/config';
import { logger } from './logger.js';
import { runPipeline } from './pipeline.js';
import { closeDb, initDb } from './db/database.js';
import { getSeasonPhase, getCurrentSeason, logSeasonStatus } from './season-manager/index.js';
import type { PipelineOptions } from './types.js';

type AlertMode = 'daily' | 'recap' | 'weekly' | 'bracket' | 'preseason' | 'season-summary' | null;

function parseArgs(): PipelineOptions & { help: boolean; alertMode: AlertMode; scoreDate?: string } {
  const args = process.argv.slice(2);
  const opts: PipelineOptions & { help: boolean; alertMode: AlertMode; scoreDate?: string } = {
    help: false,
    verbose: true,
    forceRefresh: false,
    alertMode: null,
    demo: false,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    switch (arg) {
      case '--help': case '-h': opts.help = true; break;
      case '--date': case '-d': opts.date = args[++i]; break;
      case '--force-refresh': case '-f': opts.forceRefresh = true; break;
      case '--quiet': case '-q': opts.verbose = false; break;
      case '--alert': case '-a': {
        const mode = args[++i];
        if (['daily', 'recap', 'weekly', 'bracket', 'preseason', 'season-summary'].includes(mode)) {
          opts.alertMode = mode as AlertMode;
        } else {
          console.error(`Unknown alert mode: "${mode}"`);
          process.exit(1);
        }
        break;
      }
      case '--score': opts.scoreDate = args[++i]; break;
      case '--demo': opts.demo = true; break;
      default:
        if (/^\d{4}-\d{2}-\d{2}$/.test(arg)) opts.date = arg;
    }
  }
  return opts;
}

function printHelp(): void {
  console.log(`
NCAAM Oracle v4.1 — NCAA Men's Basketball Prediction Engine
============================================================

USAGE:
  npm start [options]

OPTIONS:
  --date, -d YYYY-MM-DD        Predict for specific date (default: today)
  --alert, -a <mode>           Send Discord alert:
                                 daily         → daily predictions embed
                                 weekly        → Monday recap embed
                                 bracket       → March Madness bracket embed
                                 preseason     → season online notification
                                 season-summary→ final season report
  --score YYYY-MM-DD           Score games from that date and update DB
  --force-refresh, -f          Bypass cache, re-fetch all data
  --quiet, -q                  Suppress prediction table
  --help, -h                   Show this help

SCRIPTS:
  npm run alerts:daily          Daily predictions → Discord
  npm run alerts:weekly         Weekly recap → Discord
  npm run alerts:bracket        Bracket sim → Discord
  npm run train                 Train ML model (Python required)
  npm run fetch-historical      Fetch historical data (Python)

ENVIRONMENT (.env):
  DISCORD_WEBHOOK_URL           Discord webhook URL
  ODDS_API_KEY                  The Odds API key (optional)
  LOG_LEVEL                     Logging level (default: info)
`);
}

// ─── Alert runners ────────────────────────────────────────────────────────────

async function runDailyAlert(date: string, demo = false): Promise<void> {
  const { sendDailyPredictions } = await import('./alerts/discord.js');
  const { writePredictionsFile } = await import('./kalshi/predictionsFile.js');
  const { getPredictionsByDate } = await import('./db/database.js');
  const { fetchSchedule } = await import('./api/cbbClient.js');
  await initDb();

  // Pre-flight: if no games are scheduled today, skip Discord entirely.
  // This guards against the daily workflow firing during the offseason or on
  // empty schedule days (e.g. the day after the championship).
  if (!demo) {
    try {
      const games = await fetchSchedule(date);
      const playable = games.filter(g => g.status === 'Scheduled' || g.status === 'Live' || g.status === 'Final');
      if (playable.length === 0) {
        logger.info({ date }, 'no games today — skipping Discord');
        return;
      }
    } catch (err) {
      logger.warn({ err, date }, 'Schedule pre-flight failed — continuing');
    }
  }

  await runPipeline({ date, verbose: false, demo });
  try {
    const preds = getPredictionsByDate(date);
    if (preds.length === 0) {
      logger.info({ date }, 'no predictions to send — skipping Discord');
      return;
    }
    const path = writePredictionsFile(date, preds);
    logger.info({ path, count: preds.length }, 'Wrote kalshi-safety predictions JSON');
  } catch (err) {
    logger.warn({ err }, 'Failed to write predictions JSON (continuing)');
  }
  const ok = await sendDailyPredictions(date);
  logger.info({ ok, date }, 'Daily alert sent');
}

async function runRecapAlert(date: string): Promise<void> {
  const { sendEveningRecap } = await import('./alerts/discord.js');
  await initDb();
  const ok = await sendEveningRecap(date);
  logger.info({ ok, date }, 'Evening recap sent');
}

async function runWeeklyAlert(date: string): Promise<void> {
  const { sendWeeklyRecap } = await import('./alerts/discord.js');
  await initDb();
  // Get the start of the past week (Monday)
  const d = new Date(date);
  const dayOfWeek = d.getDay();
  const monday = new Date(d);
  monday.setDate(d.getDate() - ((dayOfWeek + 6) % 7) - 7);
  const weekStart = monday.toISOString().split('T')[0];
  const ok = await sendWeeklyRecap(weekStart);
  logger.info({ ok, weekStart }, 'Weekly recap sent');
}

async function runBracketAlert(): Promise<void> {
  const { sendBracketEmbed } = await import('./alerts/discord.js');
  await initDb();
  const season = getCurrentSeason();
  // Note: bracket simulation should have been run separately (runBracketSimulation)
  const ok = await sendBracketEmbed(season);
  logger.info({ ok, season }, 'Bracket embed sent');
}

async function runPreseasonAlert(): Promise<void> {
  const { sendPreseasonOnline } = await import('./alerts/discord.js');
  await initDb();
  const season = getCurrentSeason();
  const ok = await sendPreseasonOnline(season);
  logger.info({ ok, season }, 'Preseason online alert sent');
}

async function runSeasonSummaryAlert(): Promise<void> {
  const { sendSeasonSummary } = await import('./alerts/discord.js');
  await initDb();
  const season = getCurrentSeason();
  const ok = await sendSeasonSummary(season);
  logger.info({ ok, season }, 'Season summary sent');
}

async function runScorer(date: string): Promise<void> {
  const { scoreYesterdayGames } = await import('./alerts/results.js');
  await initDb();
  const result = await scoreYesterdayGames(date);
  console.log(`\nScoring ${date}: ${result.correct}/${result.scored} correct (${(result.accuracy * 100).toFixed(1)}%) | Brier: ${result.brierScore.toFixed(4)}\n`);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const opts = parseArgs();

  if (opts.help) { printHelp(); process.exit(0); }

  if (opts.date && !/^\d{4}-\d{2}-\d{2}$/.test(opts.date)) {
    logger.error({ date: opts.date }, 'Invalid date format (YYYY-MM-DD required)');
    process.exit(1);
  }

  const date = opts.date ?? new Date().toISOString().split('T')[0];

  logSeasonStatus(date);
  logger.info({ date, alertMode: opts.alertMode ?? 'pipeline', version: '4.1.0' }, 'NCAAM Oracle starting');

  try {
    if (opts.scoreDate) {
      await runScorer(opts.scoreDate);
      return;
    }

    switch (opts.alertMode) {
      case 'daily':          await runDailyAlert(date, opts.demo);  return;
      case 'recap':          await runRecapAlert(date);             return;
      case 'weekly':         await runWeeklyAlert(date);            return;
      case 'bracket':        await runBracketAlert();           return;
      case 'preseason':      await runPreseasonAlert();         return;
      case 'season-summary': await runSeasonSummaryAlert();     return;
    }

    // Default: run pipeline and print predictions
    if (opts.forceRefresh) {
      const { readdirSync, unlinkSync } = await import('fs');
      try {
        const files = readdirSync('cache');
        for (const f of files) if (f.endsWith('.json')) unlinkSync(`cache/${f}`);
        logger.info({ cleared: files.length }, 'Cache cleared');
      } catch { /* cache dir may not exist */ }
    }

    const predictions = await runPipeline(opts);

    if (predictions.length === 0) {
      console.log(`\nNo games scheduled for ${date} (or outside season window).\n`);
    }

  } catch (err) {
    logger.error({ err }, 'Fatal error');
    process.exit(1);
  } finally {
    closeDb();
  }
}

process.on('unhandledRejection', reason => {
  logger.error({ reason }, 'Unhandled rejection');
  process.exit(1);
});

process.on('uncaughtException', err => {
  logger.error({ err }, 'Uncaught exception');
  closeDb();
  process.exit(1);
});

main();
