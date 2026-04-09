// NCAAM Oracle v4.1 — Discord Webhook Alert Module
// Embeds:
//   Daily Predictions  — orange (#E04E39)
//   Weekly Recap       — green (#2ECC71) / red (#E74C3C)
//   March Madness      — gold (#FFD700)
//   Season Summary     — orange (#E04E39)
//   Preseason Online   — blue (#3498DB)

import fetch from 'node-fetch';
import { logger } from '../logger.js';
import {
  getPredictionsByDate,
  getSeasonRecord,
  getWeeklyRecords,
  getTournamentSims,
} from '../db/database.js';
import { getConfidenceTier, confidenceEmoji } from '../features/marketEdge.js';
import { getCurrentSeason } from '../season-manager/index.js';
import type { Prediction, TournamentSim } from '../types.js';

// ─── Colors ───────────────────────────────────────────────────────────────────

const COLORS = {
  daily:           0xE04E39,  // orange
  weekly_good:     0x2ECC71,  // green
  weekly_bad:      0xE74C3C,  // red
  weekly_neutral:  0x95A5A6,  // gray
  tournament:      0xFFD700,  // gold
  season_summary:  0xE04E39,  // orange
  preseason:       0x3498DB,  // blue
} as const;

// ─── Discord types ────────────────────────────────────────────────────────────

interface DiscordField {
  name: string;
  value: string;
  inline?: boolean;
}

interface DiscordEmbed {
  title?: string;
  description?: string;
  color?: number;
  fields?: DiscordField[];
  footer?: { text: string };
  timestamp?: string;
}

interface DiscordPayload {
  content?: string;
  embeds: DiscordEmbed[];
}

// ─── Webhook sender ───────────────────────────────────────────────────────────

async function sendWebhook(payload: DiscordPayload): Promise<boolean> {
  const webhookUrl = process.env.DISCORD_WEBHOOK_URL;
  if (!webhookUrl) {
    logger.warn('DISCORD_WEBHOOK_URL not set — skipping Discord alert');
    return false;
  }

  try {
    const resp = await fetch(webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(12000),
    });

    if (!resp.ok) {
      const text = await resp.text();
      logger.error({ status: resp.status, body: text }, 'Discord webhook error');
      return false;
    }

    logger.info('Discord alert sent successfully');
    return true;
  } catch (err) {
    logger.error({ err }, 'Failed to send Discord webhook');
    return false;
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function pct(prob: number): string {
  return (prob * 100).toFixed(1) + '%';
}

function getPickedTeam(pred: Prediction): { team: string; prob: number } {
  if (pred.calibrated_prob >= 0.5) {
    return { team: pred.home_team, prob: pred.calibrated_prob };
  }
  return { team: pred.away_team, prob: 1 - pred.calibrated_prob };
}

function spreadLabel(spread: number, homeTeam: string, awayTeam: string): string {
  if (Math.abs(spread) < 0.5) return 'PK';
  if (spread > 0) return `${homeTeam} -${Math.abs(spread).toFixed(1)}`;
  return `${awayTeam} -${Math.abs(spread).toFixed(1)}`;
}

function isHighConviction(pred: Prediction): boolean {
  const p = Math.max(pred.calibrated_prob, 1 - pred.calibrated_prob);
  return p >= 0.70;
}

function isExtremeConviction(pred: Prediction): boolean {
  const p = Math.max(pred.calibrated_prob, 1 - pred.calibrated_prob);
  return p >= 0.75;
}

function edgeStr(edge?: number): string {
  if (edge === undefined || Math.abs(edge) < 0.02) return '';
  const sign = edge > 0 ? '+' : '';
  return ` | Edge: ${sign}${(edge * 100).toFixed(1)}%`;
}

function weekTrend(weeks: Array<{ weekStart: string; accuracy: number }>): string {
  if (weeks.length === 0) return 'No data';
  return weeks
    .map((w, i) => `W${i + 1}: ${(w.accuracy * 100).toFixed(0)}%`)
    .join(' | ');
}

// ─── DAILY PREDICTIONS EMBED ─────────────────────────────────────────────────

export async function sendDailyPredictions(date: string): Promise<boolean> {
  const predictions = getPredictionsByDate(date);
  const season = getCurrentSeason();
  const record  = getSeasonRecord(season);

  if (predictions.length === 0) {
    logger.warn({ date }, 'No predictions to send');
    return false;
  }

  // Sort: extreme conviction first, then high, then others
  const sorted = [...predictions].sort((a, b) => {
    const pA = Math.abs(a.calibrated_prob - 0.5);
    const pB = Math.abs(b.calibrated_prob - 0.5);
    return pB - pA;
  });

  const highConv = sorted.filter(isHighConviction);
  const strong   = sorted.filter(p => !isHighConviction(p) && Math.max(p.calibrated_prob, 1 - p.calibrated_prob) >= 0.63);
  const leans    = sorted.filter(p => !isHighConviction(p) && Math.max(p.calibrated_prob, 1 - p.calibrated_prob) < 0.63);

  const fields: DiscordField[] = [];

  // Season record inline fields
  const seasonPct = record.total > 0 ? (record.correct / record.total * 100).toFixed(1) + '%' : 'N/A';
  const hcPct     = record.highConvTotal > 0 ? (record.highConvCorrect / record.highConvTotal * 100).toFixed(1) + '%' : 'N/A';
  const atsPct    = (record.atsWins + record.atsLosses) > 0
    ? ((record.atsWins / (record.atsWins + record.atsLosses)) * 100).toFixed(1) + '%'
    : 'N/A';

  fields.push(
    { name: '🏆 Season Record', value: `${record.correct}-${record.total - record.correct} (${seasonPct})`, inline: true },
    { name: '🎯 High-Conviction', value: `${record.highConvCorrect}-${record.highConvTotal - record.highConvCorrect} (${hcPct})`, inline: true },
    { name: '🎰 vs Vegas ATS', value: `${record.atsWins}-${record.atsLosses} (${atsPct})`, inline: true }
  );

  // High-conviction picks (detailed)
  for (const pred of highConv.slice(0, 10)) {
    const { team, prob } = getPickedTeam(pred);
    const tier   = getConfidenceTier(pred.calibrated_prob);
    const emoji  = confidenceEmoji(tier);
    const label  = isExtremeConviction(pred) ? '🚀 EXTREME' : '🟢 HIGH CONVICTION';
    const spread = spreadLabel(pred.spread, pred.home_team, pred.away_team);
    const early  = pred.early_season_label ? `\n${pred.early_season_label}` : '';
    const edge   = edgeStr(pred.edge);

    fields.push({
      name: `${label}: ${pred.away_team} @ ${pred.home_team}`,
      value: `${emoji} **Pick: ${team}** (${pct(prob)}) | Spread: ${spread} | O/U: ${pred.total_points.toFixed(0)}${edge}${early}`,
      inline: false,
    });
  }

  // Strong picks (condensed)
  if (strong.length > 0) {
    const strongLines = strong.slice(0, 8).map(pred => {
      const { team, prob } = getPickedTeam(pred);
      return `✅ **${pred.away_team} @ ${pred.home_team}**: ${team} (${pct(prob)})`;
    });
    fields.push({
      name: `✅ Strong (63–70%) — ${strong.length} pick${strong.length !== 1 ? 's' : ''}`,
      value: strongLines.join('\n') || '—',
      inline: false,
    });
  }

  // Lean picks count only
  if (leans.length > 0) {
    fields.push({
      name: `📊 Leans (55–63%) — ${leans.length} pick${leans.length !== 1 ? 's' : ''}`,
      value: leans.slice(0, 6).map(p => {
        const { team, prob } = getPickedTeam(p);
        return `${p.away_team} @ ${p.home_team}: ${team} (${pct(prob)})`;
      }).join('\n') || '—',
      inline: false,
    });
  }

  const hasEarlySeason = sorted.some(p => p.early_season_label);
  const footerText = [
    'NCAAM Oracle v4.1',
    hasEarlySeason ? '🟡 Some games use prior-year baseline' : '',
    'Brier score tracks calibration over time',
  ].filter(Boolean).join(' | ');

  const embed: DiscordEmbed = {
    title: `🏀 NCAAM Oracle — ${date} Predictions`,
    description: [
      `**${sorted.length} D-I games today.**`,
      `${highConv.length} High Conviction · ${strong.length} Strong · ${leans.length} Lean`,
      record.total > 0 ? `Brier: **${record.brier.toFixed(3)}**` : '',
    ].filter(Boolean).join('  ·  '),
    color: COLORS.daily,
    fields: fields.slice(0, 25),
    footer: { text: footerText },
    timestamp: new Date().toISOString(),
  };

  return sendWebhook({ embeds: [embed] });
}

// ─── WEEKLY RECAP EMBED ───────────────────────────────────────────────────────

export async function sendWeeklyRecap(weekStart: string): Promise<boolean> {
  const season  = getCurrentSeason();
  const record  = getSeasonRecord(season);
  const weeklyRecords = getWeeklyRecords(season, 8);

  const thisWeek = weeklyRecords[weeklyRecords.length - 1];
  const weekCorrect = thisWeek?.correct ?? 0;
  const weekTotal   = thisWeek?.total ?? 0;
  const weekAccuracy = weekTotal > 0 ? weekCorrect / weekTotal : 0;

  const color = weekAccuracy >= 0.68 ? COLORS.weekly_good
               : weekAccuracy >= 0.55 ? COLORS.weekly_neutral
               : COLORS.weekly_bad;

  const seasonPct = record.total > 0 ? (record.correct / record.total * 100).toFixed(1) + '%' : 'N/A';
  const hcPct     = record.highConvTotal > 0 ? (record.highConvCorrect / record.highConvTotal * 100).toFixed(1) + '%' : 'N/A';

  const weekNum = weeklyRecords.length;

  const trend = weekTrend(weeklyRecords.slice(-5));

  const embed: DiscordEmbed = {
    title: `📊 NCAAM Oracle — Week ${weekNum} Recap`,
    description: `Week starting ${weekStart} · Season ${season}`,
    color,
    fields: [
      {
        name: '📅 This Week',
        value: `**${weekCorrect}-${weekTotal - weekCorrect}** (${(weekAccuracy * 100).toFixed(1)}%)`,
        inline: true,
      },
      {
        name: '🏆 Season Record',
        value: `**${record.correct}-${record.total - record.correct}** (${seasonPct})`,
        inline: true,
      },
      {
        name: '🎯 High-Conviction (70%+)',
        value: `**${record.highConvCorrect}-${record.highConvTotal - record.highConvCorrect}** (${hcPct})`,
        inline: true,
      },
      {
        name: '🎰 vs Vegas ATS',
        value: `${record.atsWins}-${record.atsLosses}`,
        inline: true,
      },
      {
        name: '📉 Season Brier',
        value: record.brier.toFixed(3) + (record.brier < 0.18 ? ' ✅' : ' ⚠️'),
        inline: true,
      },
      {
        name: `📈 Last 5 Weeks`,
        value: trend,
        inline: false,
      },
    ],
    footer: { text: `NCAAM Oracle v4.1 | Week ${weekNum} of season` },
    timestamp: new Date().toISOString(),
  };

  return sendWebhook({ embeds: [embed] });
}

// ─── MARCH MADNESS BRACKET EMBED ─────────────────────────────────────────────

export async function sendBracketEmbed(season: string): Promise<boolean> {
  const sims = getTournamentSims(season);

  if (sims.length === 0) {
    logger.warn('No tournament simulations found');
    return false;
  }

  // Group by region
  const regions = ['East', 'West', 'South', 'Midwest'];
  const byRegion = new Map<string, TournamentSim[]>();
  for (const sim of sims) {
    const arr = byRegion.get(sim.region) ?? [];
    arr.push(sim);
    byRegion.set(sim.region, arr);
  }

  const fields: DiscordField[] = [];

  // Top 8 championship contenders
  const top8 = sims.slice(0, 8);
  const champLines = top8
    .map(s => `**${s.team_name}** (#${s.seed}) ${(s.champ_prob * 100).toFixed(1)}%`)
    .join('\n');
  fields.push({ name: '🏆 Championship Favorites', value: champLines, inline: false });

  // Final Four odds
  const f4 = sims.filter(s => s.f4_prob >= 0.10).slice(0, 12);
  const f4Lines = f4.map(s =>
    `${s.team_name} (${s.seed}): FF ${(s.f4_prob * 100).toFixed(0)}% | E8 ${(s.e8_prob * 100).toFixed(0)}%`
  ).join('\n');
  fields.push({ name: '🏀 Final Four Contenders (10%+ odds)', value: f4Lines || 'Computing...', inline: false });

  // Biggest upset threats (seed 10+, >15% Sweet 16)
  const upsets = sims.filter(s => s.seed >= 10 && s.s16_prob >= 0.15).slice(0, 6);
  if (upsets.length > 0) {
    const upsetLines = upsets.map(s =>
      `**(${s.seed}) ${s.team_name}**: Sweet 16 ${(s.s16_prob * 100).toFixed(0)}%`
    ).join('\n');
    fields.push({ name: '🚨 Upset Alerts (Seed 10+)', value: upsetLines, inline: false });
  }

  // Region breakdown
  for (const region of regions) {
    const regionTeams = (byRegion.get(region) ?? []).slice(0, 4);
    if (regionTeams.length === 0) continue;

    const lines = regionTeams.map(s =>
      `(${s.seed}) ${s.team_name}: ${(s.f4_prob * 100).toFixed(0)}% F4 | ${(s.champ_prob * 100).toFixed(1)}% Champ`
    ).join('\n');

    fields.push({ name: `📍 ${region} Region`, value: lines, inline: true });
  }

  const embed: DiscordEmbed = {
    title: `🏆 NCAAM Oracle — ${season} March Madness Bracket`,
    description: `Based on 10,000 bracket simulations · 68 teams · Selection Sunday ${new Date().toLocaleDateString()}`,
    color: COLORS.tournament,
    fields: fields.slice(0, 25),
    footer: { text: 'NCAAM Oracle v4.1 | 10,000 bracket simulations | Probabilities shift with each round' },
    timestamp: new Date().toISOString(),
  };

  return sendWebhook({ embeds: [embed] });
}

// ─── SEASON SUMMARY EMBED ─────────────────────────────────────────────────────

export async function sendSeasonSummary(season: string): Promise<boolean> {
  const record = getSeasonRecord(season);
  const weeklyRecords = getWeeklyRecords(season, 99);

  const seasonPct  = record.total > 0 ? (record.correct / record.total * 100).toFixed(1) + '%' : 'N/A';
  const hcPct      = record.highConvTotal > 0 ? (record.highConvCorrect / record.highConvTotal * 100).toFixed(1) + '%' : 'N/A';

  // Best and worst weeks
  let bestWeek = weeklyRecords[0];
  let worstWeek = weeklyRecords[0];
  for (const w of weeklyRecords) {
    if (w.total > 0 && w.accuracy > (bestWeek?.accuracy ?? 0)) bestWeek = w;
    if (w.total > 0 && w.accuracy < (worstWeek?.accuracy ?? 1)) worstWeek = w;
  }

  const weekBreakdown = weeklyRecords
    .map((w, i) => `W${i + 1}: ${(w.accuracy * 100).toFixed(0)}% (${w.correct}/${w.total})`)
    .join(' | ');

  const embed: DiscordEmbed = {
    title: `🏆 NCAAM Oracle — ${season} Season Final Report`,
    color: COLORS.season_summary,
    fields: [
      { name: '🏆 Final Record', value: `**${record.correct}-${record.total - record.correct}** (${seasonPct})`, inline: true },
      { name: '🎯 High-Conviction (70%+)', value: `${record.highConvCorrect}-${record.highConvTotal - record.highConvCorrect} (${hcPct})`, inline: true },
      { name: '🎰 vs Vegas ATS', value: `${record.atsWins}-${record.atsLosses}`, inline: true },
      { name: '📉 Season Brier', value: record.brier.toFixed(4), inline: true },
      { name: '🏅 Best Week', value: bestWeek ? `${bestWeek.weekStart}: ${(bestWeek.accuracy * 100).toFixed(0)}%` : 'N/A', inline: true },
      { name: '📉 Worst Week', value: worstWeek ? `${worstWeek.weekStart}: ${(worstWeek.accuracy * 100).toFixed(0)}%` : 'N/A', inline: true },
      { name: '📊 Total Games Predicted', value: String(record.total), inline: true },
      { name: '📈 Week-by-Week', value: weekBreakdown || 'N/A', inline: false },
    ],
    footer: { text: 'Season complete. See you in October! | NCAAM Oracle v4.1' },
    timestamp: new Date().toISOString(),
  };

  return sendWebhook({ embeds: [embed] });
}

// ─── PRESEASON ONLINE EMBED ───────────────────────────────────────────────────

export async function sendPreseasonOnline(season: string): Promise<boolean> {
  const embed: DiscordEmbed = {
    title: `🏀 NCAAM Oracle is ONLINE for ${season}!`,
    description: [
      'The prediction engine has been initialized for the new season.',
      '',
      '**What to expect:**',
      '• Daily predictions at 9 AM ET (Nov–Apr)',
      '• Monday weekly recaps',
      '• March Madness bracket simulation on Selection Sunday',
      '• High-conviction picks (70%+) and edge detection vs Vegas',
    ].join('\n'),
    color: COLORS.preseason,
    fields: [
      { name: '📅 Season', value: season, inline: true },
      { name: '🎯 Target Accuracy', value: '68–72% overall | 75–78% at 70%+', inline: true },
      { name: '📊 Model', value: 'Logistic Regression + Monte Carlo (10k sims)', inline: false },
      { name: '📡 Data Sources', value: 'BartTorvik · ESPN · Odds API · 247Sports', inline: false },
    ],
    footer: { text: 'NCAAM Oracle v4.1 | GitHub Actions + Discord Embeds' },
    timestamp: new Date().toISOString(),
  };

  return sendWebhook({ embeds: [embed] });
}

// ─── TOURNAMENT DAY RESULTS EMBED ────────────────────────────────────────────

export async function sendTournamentRoundResults(
  round: string,
  results: Array<{ prediction: Prediction; homeScore: number; awayScore: number }>
): Promise<boolean> {
  const correct = results.filter(r => r.prediction.correct).length;
  const total   = results.length;
  const accuracy = total > 0 ? correct / total : 0;

  const lines = results.map(({ prediction: p, homeScore, awayScore }) => {
    const { team } = getPickedTeam(p);
    const isCorrect = p.correct ? '✅' : '❌';
    return `${isCorrect} **${p.away_team} @ ${p.home_team}**: ${awayScore}–${homeScore} *(picked ${team})*`;
  });

  const embed: DiscordEmbed = {
    title: `🏆 March Madness — ${round} Results`,
    description: `**${correct}/${total} correct** (${(accuracy * 100).toFixed(0)}%)`,
    color: accuracy >= 0.65 ? COLORS.weekly_good : accuracy >= 0.50 ? COLORS.weekly_neutral : COLORS.weekly_bad,
    fields: [
      {
        name: '🎯 Game-by-Game',
        value: lines.join('\n') || 'No results',
        inline: false,
      },
    ],
    footer: { text: 'NCAAM Oracle v4.1 | March Madness Mode' },
    timestamp: new Date().toISOString(),
  };

  return sendWebhook({ embeds: [embed] });
}
