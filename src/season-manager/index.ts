// NCAAM Oracle v4.1 — Season Lifecycle Manager
// Handles date-aware gating: November through April
// Detects March Madness mode (Selection Sunday → Championship)

import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import { logger } from '../logger.js';

interface SeasonConfig {
  season: string;             // e.g. "2025-26"
  season_start: string;       // YYYY-MM-DD (early November)
  season_end: string;         // YYYY-MM-DD (day after Championship, early April)
  conference_tourney_start: string;
  selection_sunday: string;
  tournament_start: string;
  championship: string;
}

const CONFIG_PATH = resolve('config/season.json');

function loadConfig(): SeasonConfig {
  if (!existsSync(CONFIG_PATH)) {
    // Default config for 2025-26 season
    return {
      season: '2025-26',
      season_start: '2025-11-03',
      season_end: '2026-04-08',
      conference_tourney_start: '2026-03-03',
      selection_sunday: '2026-03-15',
      tournament_start: '2026-03-17',
      championship: '2026-04-06',
    };
  }
  return JSON.parse(readFileSync(CONFIG_PATH, 'utf-8')) as SeasonConfig;
}

export type SeasonPhase =
  | 'dormant'
  | 'early_season'
  | 'conference_play'
  | 'conference_tourney'
  | 'march_madness'
  | 'season_end';

export function getSeasonPhase(dateStr?: string): SeasonPhase {
  const cfg = loadConfig();
  const date = dateStr ? new Date(dateStr) : new Date();
  const d = date.toISOString().split('T')[0];

  if (d < cfg.season_start || d > cfg.season_end) return 'dormant';

  // Day after championship → season_end (send final summary)
  if (d > cfg.championship && d <= cfg.season_end) return 'season_end';

  // March Madness (Selection Sunday → Championship)
  if (d >= cfg.selection_sunday && d <= cfg.championship) return 'march_madness';

  // Conference tournaments
  if (d >= cfg.conference_tourney_start && d < cfg.selection_sunday) return 'conference_tourney';

  // Early season: November through end of December
  const month = date.getMonth() + 1; // 1-indexed
  if (month === 11 || month === 12) return 'early_season';

  return 'conference_play';
}

export function isInSeason(dateStr?: string): boolean {
  const phase = getSeasonPhase(dateStr);
  return phase !== 'dormant';
}

export function isTournamentMode(dateStr?: string): boolean {
  const phase = getSeasonPhase(dateStr);
  return phase === 'march_madness';
}

export function isSelectionSunday(dateStr?: string): boolean {
  const cfg = loadConfig();
  const d = dateStr ?? new Date().toISOString().split('T')[0];
  return d === cfg.selection_sunday;
}

export function isChampionshipDay(dateStr?: string): boolean {
  const cfg = loadConfig();
  const d = dateStr ?? new Date().toISOString().split('T')[0];
  return d === cfg.championship;
}

export function getCurrentSeason(): string {
  return loadConfig().season;
}

export function getSeasonConfig(): SeasonConfig {
  return loadConfig();
}

export function phaseLabel(phase: SeasonPhase): string {
  const labels: Record<SeasonPhase, string> = {
    dormant: 'Offseason',
    early_season: 'Early Season',
    conference_play: 'Conference Play',
    conference_tourney: 'Conference Tournaments',
    march_madness: 'March Madness',
    season_end: 'Season Complete',
  };
  return labels[phase];
}

export function logSeasonStatus(dateStr?: string): void {
  const phase = getSeasonPhase(dateStr);
  const d = dateStr ?? new Date().toISOString().split('T')[0];
  logger.info({ date: d, phase, label: phaseLabel(phase) }, 'Season status');

  if (phase === 'dormant') {
    logger.info('Outside active season window — pipeline will exit');
  } else if (phase === 'march_madness') {
    logger.info('March Madness mode active — tournament adjustments enabled');
  }
}
