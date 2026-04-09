// NCAAM Oracle v4.1 — Vegas Odds Client
// Uses The Odds API (free tier: 500 requests/month)
// Falls back to manual vegas_lines.json if API key not set.

import { existsSync, readFileSync } from 'fs';
import { resolve } from 'path';
import fetch from 'node-fetch';
import { logger } from '../logger.js';

const MANUAL_LINES_PATH = resolve('data/vegas_lines.json');
const ODDS_API_KEY = process.env.ODDS_API_KEY ?? '';

interface ManualLines {
  [date: string]: {
    [matchupKey: string]: { homeML: number; awayML: number };
  };
}

interface OddsApiGame {
  id: string;
  home_team: string;
  away_team: string;
  bookmakers: Array<{
    key: string;
    markets: Array<{
      key: string;
      outcomes: Array<{ name: string; price: number }>;
    }>;
  }>;
}

export function mlToImplied(ml: number): number {
  if (ml > 0) return 100 / (ml + 100);
  return Math.abs(ml) / (Math.abs(ml) + 100);
}

export function removeVig(homeML: number, awayML: number): { homeProb: number; awayProb: number } {
  const rawHome = mlToImplied(homeML);
  const rawAway = mlToImplied(awayML);
  const total = rawHome + rawAway;
  return { homeProb: rawHome / total, awayProb: rawAway / total };
}

let _oddsMap: Map<string, { homeImpliedProb: number; awayImpliedProb: number; homeML: number; awayML: number }> | null = null;

export function getOddsForGame(matchupKey: string): { homeImpliedProb: number; awayImpliedProb: number; homeML: number; awayML: number } | null {
  return _oddsMap?.get(matchupKey) ?? null;
}

export function hasAnyOdds(): boolean {
  return (_oddsMap?.size ?? 0) > 0;
}

function loadManualLines(date: string): Map<string, { homeML: number; awayML: number }> {
  if (!existsSync(MANUAL_LINES_PATH)) return new Map();
  try {
    const data = JSON.parse(readFileSync(MANUAL_LINES_PATH, 'utf-8')) as ManualLines;
    const dayLines = data[date] ?? {};
    const map = new Map<string, { homeML: number; awayML: number }>();
    for (const [key, val] of Object.entries(dayLines)) {
      map.set(key, val);
    }
    return map;
  } catch {
    return new Map();
  }
}

export async function loadOddsApiLines(date: string): Promise<Map<string, { homeML: number; awayML: number }>> {
  if (!ODDS_API_KEY) return new Map();

  try {
    const url = `https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds/?apiKey=${ODDS_API_KEY}&regions=us&markets=h2h&dateFormat=iso&oddsFormat=american`;
    const resp = await fetch(url, { signal: AbortSignal.timeout(10000) });
    if (!resp.ok) {
      logger.warn({ status: resp.status }, 'Odds API returned error');
      return new Map();
    }

    const games = await resp.json() as OddsApiGame[];
    const map = new Map<string, { homeML: number; awayML: number }>();

    for (const game of games) {
      const bookmaker = game.bookmakers.find(b => b.key === 'draftkings') ??
                        game.bookmakers.find(b => b.key === 'fanduel') ??
                        game.bookmakers[0];
      if (!bookmaker) continue;

      const h2h = bookmaker.markets.find(m => m.key === 'h2h');
      if (!h2h) continue;

      const homeOutcome = h2h.outcomes.find(o => o.name === game.home_team);
      const awayOutcome = h2h.outcomes.find(o => o.name === game.away_team);
      if (!homeOutcome || !awayOutcome) continue;

      const key = `${abbreviateTeam(game.away_team)}@${abbreviateTeam(game.home_team)}`;
      map.set(key, { homeML: homeOutcome.price, awayML: awayOutcome.price });
    }

    logger.info({ games: map.size, date }, 'Odds API lines loaded');
    return map;
  } catch (err) {
    logger.warn({ err }, 'Failed to load Odds API lines');
    return new Map();
  }
}

export async function initializeOdds(date: string): Promise<void> {
  _oddsMap = new Map();

  // Try Odds API first
  const apiLines = await loadOddsApiLines(date);

  // Fall back to manual lines
  const manualLines = loadManualLines(date);

  const allLines = new Map([...manualLines, ...apiLines]);

  for (const [key, { homeML, awayML }] of allLines) {
    const { homeProb, awayProb } = removeVig(homeML, awayML);
    _oddsMap.set(key, {
      homeImpliedProb: homeProb,
      awayImpliedProb: awayProb,
      homeML,
      awayML,
    });
  }

  logger.info({ games: _oddsMap.size }, 'Odds initialized');
}

function abbreviateTeam(fullName: string): string {
  // Rough mapping — real implementation would use ESPN team data
  return fullName.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 4);
}
