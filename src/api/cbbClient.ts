// NCAAM Oracle v4.1 — CBB Data Client
// Primary sources:
//   ESPN public API — schedule, scores, team info (no key required)
//   BartTorvik.com  — AdjEM, AdjOE, AdjDE, AdjTempo, BARTHAG, WAB (free)
// KenPom is ~$25/yr — BartTorvik is used as the free alternative.

import { mkdirSync, readFileSync, writeFileSync, existsSync, statSync } from 'fs';
import { resolve } from 'path';
import fetch from 'node-fetch';
import { logger } from '../logger.js';
import type { CBBTeam, CBBGame, PriorYearData } from '../types.js';

const CACHE_DIR = process.env.CACHE_DIR ?? resolve('cache');
const CACHE_TTL_MS = (Number(process.env.CACHE_TTL_HOURS ?? 4)) * 60 * 60 * 1000;

mkdirSync(CACHE_DIR, { recursive: true });

// ─── ESPN API base URLs ───────────────────────────────────────────────────────

const ESPN_BASE = 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball';
const ESPN_WEB  = 'https://site.web.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball';

// ─── BartTorvik API ───────────────────────────────────────────────────────────

// ─── Cache helpers ────────────────────────────────────────────────────────────

function cacheKey(url: string): string {
  return url.replace(/[^a-zA-Z0-9]/g, '_').slice(0, 180) + '.json';
}

function readCache<T>(key: string): T | null {
  const path = resolve(CACHE_DIR, key);
  if (!existsSync(path)) return null;
  const stat = statSync(path);
  if (Date.now() - stat.mtimeMs > CACHE_TTL_MS) return null;
  try { return JSON.parse(readFileSync(path, 'utf-8')) as T; }
  catch { return null; }
}

function writeCache(key: string, data: unknown): void {
  try { writeFileSync(resolve(CACHE_DIR, key), JSON.stringify(data), 'utf-8'); }
  catch (err) { logger.warn({ err }, 'Cache write failed'); }
}

async function fetchWithRetry<T>(url: string, retries = 3, delayMs = 1200): Promise<T> {
  const key = cacheKey(url);
  const cached = readCache<T>(key);
  if (cached !== null) {
    logger.debug({ url }, 'Cache hit');
    return cached;
  }

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const resp = await fetch(url, {
        headers: { 'User-Agent': 'NCAAM-Oracle/4.1' },
        signal: AbortSignal.timeout(15000),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
      const data = await resp.json() as T;
      writeCache(key, data);
      return data;
    } catch (err) {
      logger.warn({ err, attempt, url }, 'Fetch failed');
      if (attempt < retries) await sleep(delayMs * attempt);
    }
  }
  throw new Error(`Failed to fetch after ${retries} retries: ${url}`);
}

function sleep(ms: number): Promise<void> {
  return new Promise(res => setTimeout(res, ms));
}

// ─── ESPN: Today's schedule ───────────────────────────────────────────────────

interface ESPNScoreboard {
  events: ESPNEvent[];
}

interface ESPNEvent {
  id: string;
  date: string;
  name: string;
  status: { type: { description: string; completed: boolean } };
  competitions: ESPNCompetition[];
}

interface ESPNCompetition {
  id: string;
  venue?: { fullName?: string };
  neutralSite: boolean;
  notes?: Array<{ headline?: string }>;
  competitors: ESPNCompetitor[];
}

interface ESPNCompetitor {
  id: string;
  team: { id: string; abbreviation: string; displayName: string; location: string };
  homeAway: 'home' | 'away';
  score?: string;
}

export async function fetchSchedule(dateStr: string): Promise<CBBGame[]> {
  const dateFormatted = dateStr.replace(/-/g, '');
  const url = `${ESPN_BASE}/scoreboard?dates=${dateFormatted}&groups=50&limit=200`;

  let data: ESPNScoreboard;
  try {
    data = await fetchWithRetry<ESPNScoreboard>(url);
  } catch (err) {
    logger.error({ err, date: dateStr }, 'Failed to fetch schedule');
    return [];
  }

  const games: CBBGame[] = [];

  for (const event of data.events ?? []) {
    const comp = event.competitions[0];
    if (!comp) continue;

    const homeComp = comp.competitors.find(c => c.homeAway === 'home');
    const awayComp = comp.competitors.find(c => c.homeAway === 'away');

    if (!homeComp || !awayComp) continue;

    const statusDesc = event.status.type.description;
    const isFinal = event.status.type.completed;
    const isTournament = (comp.notes ?? []).some(n =>
      n.headline?.toLowerCase().includes('ncaa') ||
      n.headline?.toLowerCase().includes('tournament')
    );

    games.push({
      gameId: event.id,
      gameDate: dateStr,
      gameTime: event.date,
      status: isFinal ? 'Final' : statusDesc === 'In Progress' ? 'Live' : 'Scheduled',
      homeTeam: {
        teamId: Number(homeComp.team.id),
        teamAbbr: normalizeAbbr(homeComp.team.abbreviation),
        teamName: homeComp.team.displayName,
        score: homeComp.score ? Number(homeComp.score) : undefined,
      },
      awayTeam: {
        teamId: Number(awayComp.team.id),
        teamAbbr: normalizeAbbr(awayComp.team.abbreviation),
        teamName: awayComp.team.displayName,
        score: awayComp.score ? Number(awayComp.score) : undefined,
      },
      arena: comp.venue?.fullName ?? 'Unknown Arena',
      isNeutralSite: comp.neutralSite,
      isTournamentGame: isTournament,
    });
  }

  logger.info({ date: dateStr, games: games.length }, 'Schedule fetched');
  return games;
}

// ─── ESPN: Team statistics (primary stats source) ────────────────────────────
// BartTorvik blocks server-side requests — ESPN is the reliable alternative.
// We fetch per-team stats from ESPN's statistics endpoint as needed.

interface ESPNStatCategory {
  name: string;
  stats: Array<{ name: string; displayValue: string; value: number }>;
}

// In-memory cache for team stats (keyed by ESPN abbr)
let _teamStatsCache: Map<string, CBBTeam> | null = null;
let _teamStatsCacheAt = 0;

// ─── Fetch team stats from ESPN ───────────────────────────────────────────────

async function fetchTeamStatsById(teamId: number, teamAbbr: string, teamName: string): Promise<CBBTeam | null> {
  const year = getCurrentYear();
  const url = `${ESPN_BASE}/teams/${teamId}/statistics?season=${year}`;

  try {
    const data = await fetchWithRetry<{ results?: { stats?: { categories?: ESPNStatCategory[] } } }>(url, 2, 800);
    const categories = data?.results?.stats?.categories ?? [];

    const getStat = (catName: string, statName: string): number => {
      const cat = categories.find(c => c.name === catName);
      if (!cat) return 0;
      const stat = cat.stats.find(s => s.name === statName);
      return stat?.value ?? 0;
    };

    const gp = getStat('general', 'gamesPlayed');
    const wins = getStat('general', 'wins') || 0;
    const losses = gp - wins;

    // Offensive stats
    const avgPts = getStat('offensive', 'avgPoints');
    const fgPct  = getStat('offensive', 'fieldGoalPct');
    const fg3Pct = getStat('offensive', 'threePointFieldGoalPct');
    const ftPct  = getStat('offensive', 'freeThrowPct');
    const avgFga = getStat('offensive', 'avgFieldGoalsAttempted');
    const avg3pa = getStat('offensive', 'avgThreePointFieldGoalsAttempted');
    const avgFta = getStat('offensive', 'avgFreeThrowsAttempted');
    const avgFga2 = Math.max(0, avgFga - avg3pa);

    // Defensive / general
    const avgOreb = getStat('general', 'avgOffensiveRebounds') || getStat('defensive', 'avgOffensiveRebounds');
    const avgDreb = getStat('defensive', 'avgDefensiveRebounds');
    const avgBlk  = getStat('defensive', 'avgBlocks');
    const avgStl  = getStat('defensive', 'avgSteals');
    const avgTov  = getStat('general', 'avgTurnovers');
    const avgFouls = getStat('general', 'avgFouls');

    // Approximate AdjEM from scoring margin (raw, not pace-adjusted)
    // For early season, this is fine — blending will mix with prior-year anyway
    const avgPtsAllowed = getStat('defensive', 'avgPoints') || getStat('general', 'avgPointsAgainst') || 68;
    const rawEM = avgPts - avgPtsAllowed;

    // Approximate efficiency metrics (not pace-adjusted without tempo data)
    const adjOE = avgPts > 0 ? (avgPts / 0.68) * 0.98 : 100; // rough poss estimate
    const adjDE = avgPtsAllowed > 0 ? (avgPtsAllowed / 0.68) * 1.02 : 100;
    const adjEM = adjOE - adjDE;

    // Four factors approximations
    const efgPct = avgFga > 0 ? (fgPct * avgFga + 0.5 * fg3Pct * avg3pa) / avgFga : 0.50;
    const ftRate  = avgFga > 0 ? avgFta / avgFga * 100 : 30;
    const tovPct  = gp > 0 && avgFga > 0 ? avgTov / (avgFga + 0.44 * avgFta + avgTov) * 100 : 18;
    const totalReb = avgOreb + avgDreb;
    const orebPct = totalReb > 0 ? (avgOreb / totalReb) * 100 : 28;

    return {
      teamId,
      teamAbbr,
      teamName,
      conference: '',
      gamesPlayed: gp,
      adjEM,
      adjOE,
      adjDE,
      adjTempo: 68,  // ESPN doesn't expose tempo directly
      barthag: Math.max(0.01, Math.min(0.99, 0.5 + adjEM / 60)),
      wab: 0,
      efgPct,
      efgAllowed: 0.50,
      tovPct,
      tovForced: 18,
      orebPct,
      drebPct: 100 - orebPct,
      ftRate,
      ftRateAllowed: 30,
      threePtPct: fg3Pct * 100,
      twoPtPct: (avgFga2 > 0 ? (fgPct * avgFga - fg3Pct * avg3pa) / avgFga2 : 0.5) * 100,
      blockPct: avgBlk,
      stealPct: avgStl,
      winPct: gp > 0 ? wins / gp : 0.5,
      sos: 0,
    };
  } catch {
    return null;
  }
}

export async function fetchAllTeamStats(
  gamesForDate?: Array<{ homeTeam: { teamId: number; teamAbbr: string; teamName: string }; awayTeam: { teamId: number; teamAbbr: string; teamName: string } }>
): Promise<Map<string, CBBTeam>> {
  if (_teamStatsCache === null) _teamStatsCache = new Map();

  // Determine which teams need stats
  const teamsNeeded = new Set<string>();
  const teamIdMap = new Map<string, { id: number; name: string }>();

  const games = gamesForDate ?? [];
  for (const game of games) {
    teamsNeeded.add(game.homeTeam.teamAbbr);
    teamsNeeded.add(game.awayTeam.teamAbbr);
    teamIdMap.set(game.homeTeam.teamAbbr, { id: game.homeTeam.teamId, name: game.homeTeam.teamName });
    teamIdMap.set(game.awayTeam.teamAbbr, { id: game.awayTeam.teamId, name: game.awayTeam.teamName });
  }

  const needed = Array.from(teamsNeeded).filter(a => !_teamStatsCache!.has(a));

  if (needed.length === 0) return _teamStatsCache;

  // Batch fetch in groups of 6 to avoid rate-limiting
  for (let i = 0; i < needed.length; i += 6) {
    const batch = needed.slice(i, i + 6);
    const results = await Promise.all(
      batch.map(abbr => {
        const info = teamIdMap.get(abbr);
        if (!info || info.id === 0) return null;
        return fetchTeamStatsById(info.id, abbr, info.name);
      })
    );

    for (let j = 0; j < batch.length; j++) {
      const stats = results[j];
      if (stats) _teamStatsCache!.set(batch[j], stats);
    }

    if (i + 6 < needed.length) await sleep(400);
  }

  logger.info({ teams: _teamStatsCache.size, loaded: needed.length }, 'ESPN team stats loaded');
  return _teamStatsCache;
}

// ─── ESPN: Recent game results for a team (rolling form) ─────────────────────

export async function fetchTeamRecentGames(
  teamId: number,
  teamAbbr: string,
  limit = 10
): Promise<{ adjEM: number; gamesPlayed: number } | null> {
  try {
    const year = getCurrentYear();
    const url = `${ESPN_BASE}/teams/${teamId}/schedule?season=${year}`;
    const data = await fetchWithRetry<{ events?: { competitions?: unknown[] }[] }>(url);
    const events = data.events ?? [];
    // Get last `limit` completed games and average margin
    const completedGames = events.filter((e: { competitions?: unknown[] }) =>
      (e.competitions?.[0] as { status?: { type?: { completed?: boolean } } } | undefined)?.status?.type?.completed
    ).slice(-limit);

    return { adjEM: 0, gamesPlayed: completedGames.length };
  } catch {
    return null;
  }
}

// ─── Prior-year bootstrap data ────────────────────────────────────────────────

let _priorYear: Map<string, PriorYearData> | null = null;

export function loadPriorYearData(): Map<string, PriorYearData> {
  if (_priorYear) return _priorYear;

  const path = resolve('config/prior_year.json');
  if (!existsSync(path)) {
    logger.warn('prior_year.json not found — using defaults for all teams');
    _priorYear = new Map();
    return _priorYear;
  }

  try {
    const raw = JSON.parse(readFileSync(path, 'utf-8')) as PriorYearData[];
    const map = new Map<string, PriorYearData>();
    for (const team of raw) {
      map.set(team.teamAbbr, team);
    }
    logger.info({ teams: map.size }, 'Prior-year data loaded');
    _priorYear = map;
    return _priorYear;
  } catch (err) {
    logger.error({ err }, 'Failed to load prior_year.json');
    _priorYear = new Map();
    return _priorYear;
  }
}

// ─── Portal impact data ───────────────────────────────────────────────────────

interface PortalImpact {
  teamAbbr: string;
  portalWAR: number;
}

let _portalImpact: Map<string, number> | null = null;

export function loadPortalImpact(): Map<string, number> {
  if (_portalImpact) return _portalImpact;

  const path = resolve('config/portal_impact.json');
  if (!existsSync(path)) {
    _portalImpact = new Map();
    return _portalImpact;
  }

  try {
    const raw = JSON.parse(readFileSync(path, 'utf-8')) as PortalImpact[];
    const map = new Map<string, number>();
    for (const t of raw) map.set(t.teamAbbr, t.portalWAR);
    _portalImpact = map;
    return _portalImpact;
  } catch {
    _portalImpact = new Map();
    return _portalImpact;
  }
}

// ─── Home court advantage by team ─────────────────────────────────────────────

// Default HCA values per conference tier (points)
// Power conferences: ~3.5–5.5 pts  |  Mid-majors: ~3.0–4.5 pts
const DEFAULT_HCA: Record<string, number> = {
  'BE': 4.5,   // Big East
  'ACC': 4.5,  // ACC
  'B10': 4.0,  // Big Ten
  'B12': 4.0,  // Big 12
  'SEC': 4.0,  // SEC
  'P12': 3.8,  // Pac-12
  'MWC': 3.5,  // Mountain West
  'AAC': 3.5,  // AAC
  'WCC': 4.0,  // West Coast (Gonzaga arena)
  'A10': 3.5,  // Atlantic 10
};

const ELITE_HOME_COURTS: Record<string, number> = {
  'KU': 6.0,   // Allen Fieldhouse
  'DUKE': 5.5, // Cameron Indoor
  'GONZ': 5.5, // Kennel
  'UK': 5.0,   // Rupp Arena
  'UNC': 5.0,  // Dean Dome
  'SYRA': 5.0, // Carrier Dome
  'IOWA': 4.5,
};

export function getHomeCourtAdvantage(teamAbbr: string, conference: string, isNeutral: boolean): number {
  if (isNeutral) return 0;
  return ELITE_HOME_COURTS[teamAbbr] ?? DEFAULT_HCA[conference] ?? 3.5;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function getCurrentYear(): number {
  const now = new Date();
  // CBB season spans calendar years: 2025-26 season uses year=2026
  return now.getMonth() >= 9 ? now.getFullYear() + 1 : now.getFullYear();
}

function normalizeAbbr(abbr: string): string {
  const fixes: Record<string, string> = {
    'CONN': 'UCONN',
    'GS': 'GSU',
    'USF': 'SFLA',
    'SC': 'SCAR',
    'MIA': 'MIA',
  };
  return fixes[abbr] ?? abbr;
}

export { getCurrentYear, normalizeAbbr };
