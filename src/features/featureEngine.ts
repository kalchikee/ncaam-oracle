// NCAAM Oracle v4.1 — Feature Engineering
// Computes 40+ features as home-vs-away differences.
// Prior-year bootstrap: blends prior-year and in-season stats based on games played.
// Blending schedule:
//   0–2 games  → 80% prior / 20% in-season
//   3–5 games  → 60% prior / 40% in-season
//   6–10 games → 40% prior / 60% in-season
//   11–15 games→ 25% prior / 75% in-season
//   16–20 games→ 15% prior / 85% in-season
//   20+ games  → 10% prior / 90% in-season
//   Tournament → 5% prior / 95% in-season

import { logger } from '../logger.js';
import type { CBBGame, CBBTeam, FeatureVector, PriorYearData } from '../types.js';
import {
  fetchAllTeamStats,
  loadPriorYearData,
  loadPortalImpact,
  getHomeCourtAdvantage,
} from '../api/cbbClient.js';
import { getEloDiff, getEloWinProb, seedElo, LEAGUE_MEAN_ELO } from './eloEngine.js';

const D1_AVG_ADJ_OE = 100.0;
const D1_AVG_ADJ_DE = 100.0;
const D1_AVG_TEMPO   = 68.0;

// ─── Prior-year blend weight ──────────────────────────────────────────────────

function blendWeight(gamesPlayed: number, isTournament = false): { prior: number; inSeason: number } {
  if (isTournament) return { prior: 0.05, inSeason: 0.95 };
  if (gamesPlayed <= 2)  return { prior: 0.80, inSeason: 0.20 };
  if (gamesPlayed <= 5)  return { prior: 0.60, inSeason: 0.40 };
  if (gamesPlayed <= 10) return { prior: 0.40, inSeason: 0.60 };
  if (gamesPlayed <= 15) return { prior: 0.25, inSeason: 0.75 };
  if (gamesPlayed <= 20) return { prior: 0.15, inSeason: 0.85 };
  return { prior: 0.10, inSeason: 0.90 };
}

function earlySeasonLabel(gamesPlayed: number): string | undefined {
  if (gamesPlayed <= 5)  return '🟡 EARLY SEASON — Prior-Year Baseline';
  if (gamesPlayed <= 10) return '🟠 Blending prior-year + current';
  return undefined;
}

// ─── Blend a single stat ──────────────────────────────────────────────────────

function blend(
  priorVal: number,
  inSeasonVal: number,
  gamesPlayed: number,
  isTournament = false
): number {
  const w = blendWeight(gamesPlayed, isTournament);
  return w.prior * priorVal + w.inSeason * inSeasonVal;
}

// ─── Rest days calculation ────────────────────────────────────────────────────

function restDays(gameDate: string, lastGameDate: string | null): number {
  if (!lastGameDate) return 3; // assume rested at season start
  const ms = new Date(gameDate).getTime() - new Date(lastGameDate).getTime();
  return Math.round(ms / 86_400_000);
}

// ─── Default stats when we have zero data ────────────────────────────────────

function defaultTeam(abbr: string): CBBTeam {
  return {
    teamId: 0,
    teamAbbr: abbr,
    teamName: abbr,
    conference: '',
    gamesPlayed: 0,
    adjEM: 0,
    adjOE: D1_AVG_ADJ_OE,
    adjDE: D1_AVG_ADJ_DE,
    adjTempo: D1_AVG_TEMPO,
    barthag: 0.5,
    wab: 0,
    efgPct: 0.50,
    efgAllowed: 0.50,
    tovPct: 18,
    tovForced: 18,
    orebPct: 28,
    drebPct: 72,
    ftRate: 30,
    ftRateAllowed: 30,
    threePtPct: 33,
    twoPtPct: 50,
    blockPct: 5,
    stealPct: 8,
    winPct: 0.5,
    sos: 0,
  };
}

function defaultPrior(abbr: string): PriorYearData {
  return {
    teamId: 0,
    teamAbbr: abbr,
    teamName: abbr,
    adjEM: 0,
    adjOE: D1_AVG_ADJ_OE,
    adjDE: D1_AVG_ADJ_DE,
    adjTempo: D1_AVG_TEMPO,
    barthag: 0.5,
    pythagorean: 0.5,
    recruitingComposite: 0.5,
    returningMinutesPct: 0.6,
    portalWAR: 0,
    experienceScore: 2.0,
  };
}

// ─── Main feature computation ─────────────────────────────────────────────────

export async function computeFeatures(
  game: CBBGame,
  gameDate: string
): Promise<FeatureVector> {
  const homeAbbr = game.homeTeam.teamAbbr;
  const awayAbbr = game.awayTeam.teamAbbr;
  const isNeutral = game.isNeutralSite;
  const isTournament = game.isTournamentGame;

  logger.debug({ home: homeAbbr, away: awayAbbr }, 'Computing features');

  // Seed Elos for any new teams
  seedElo(homeAbbr);
  seedElo(awayAbbr);

  // Get cached in-season stats (pre-populated by pipeline before feature computation)
  const teamStats = await fetchAllTeamStats();
  const priorYear = loadPriorYearData();
  const portalImpact = loadPortalImpact();

  const homeInSeason = teamStats.get(homeAbbr) ?? defaultTeam(homeAbbr);
  const awayInSeason = teamStats.get(awayAbbr) ?? defaultTeam(awayAbbr);

  const homePrior = priorYear.get(homeAbbr) ?? defaultPrior(homeAbbr);
  const awayPrior = priorYear.get(awayAbbr) ?? defaultPrior(awayAbbr);

  const homeGP = homeInSeason.gamesPlayed;
  const awayGP  = awayInSeason.gamesPlayed;

  // For features that need both teams' game count, use the minimum
  const minGP = Math.min(homeGP, awayGP);
  const homeW  = blendWeight(homeGP, isTournament);
  const awayW  = blendWeight(awayGP, isTournament);

  // ── AdjEM (blended) ──────────────────────────────────────────────────────
  const homeAdjEM = homeW.prior * homePrior.adjEM + homeW.inSeason * homeInSeason.adjEM;
  const awayAdjEM  = awayW.prior * awayPrior.adjEM  + awayW.inSeason * awayInSeason.adjEM;

  const homeAdjOE = homeW.prior * homePrior.adjOE + homeW.inSeason * homeInSeason.adjOE;
  const awayAdjOE  = awayW.prior * awayPrior.adjOE  + awayW.inSeason * awayInSeason.adjOE;

  const homeAdjDE = homeW.prior * homePrior.adjDE + homeW.inSeason * homeInSeason.adjDE;
  const awayAdjDE  = awayW.prior * awayPrior.adjDE  + awayW.inSeason * awayInSeason.adjDE;

  const homeAdjTempo = homeW.prior * homePrior.adjTempo + homeW.inSeason * homeInSeason.adjTempo;
  const awayAdjTempo  = awayW.prior * awayPrior.adjTempo  + awayW.inSeason * awayInSeason.adjTempo;

  const homeBarthag = homeW.prior * homePrior.barthag + homeW.inSeason * homeInSeason.barthag;
  const awayBarthag  = awayW.prior * awayPrior.barthag  + awayW.inSeason * awayInSeason.barthag;

  // ── Pythagorean ──────────────────────────────────────────────────────────
  const homePythagorean = homeW.prior * homePrior.pythagorean + homeW.inSeason * homeInSeason.winPct;
  const awayPythagorean  = awayW.prior * awayPrior.pythagorean  + awayW.inSeason * awayInSeason.winPct;

  // ── Recruiting / roster (prior-year only) ────────────────────────────────
  const homeRecruiting = homePrior.recruitingComposite;
  const awayRecruiting  = awayPrior.recruitingComposite;

  const homeReturning = homePrior.returningMinutesPct;
  const awayReturning  = awayPrior.returningMinutesPct;

  const homePortalWAR = portalImpact.get(homeAbbr) ?? homePrior.portalWAR;
  const awayPortalWAR  = portalImpact.get(awayAbbr)  ?? awayPrior.portalWAR;

  const homeExperience = homePrior.experienceScore;
  const awayExperience  = awayPrior.experienceScore;

  // ── In-season shooting (use blend, default to 0 diff when very early) ───
  const homeEfg = blend(0.50, homeInSeason.efgPct,     homeGP, isTournament);
  const awayEfg  = blend(0.50, awayInSeason.efgPct,      awayGP, isTournament);

  const homeEfgAllowed = blend(0.50, homeInSeason.efgAllowed, homeGP, isTournament);
  const awayEfgAllowed  = blend(0.50, awayInSeason.efgAllowed,  awayGP, isTournament);

  const homeTov = blend(18, homeInSeason.tovPct, homeGP, isTournament);
  const awayTov  = blend(18, awayInSeason.tovPct,  awayGP, isTournament);

  const homeTovForced = blend(18, homeInSeason.tovForced, homeGP, isTournament);
  const awayTovForced  = blend(18, awayInSeason.tovForced,  awayGP, isTournament);

  const homeOreb = blend(28, homeInSeason.orebPct, homeGP, isTournament);
  const awayOreb  = blend(28, awayInSeason.orebPct,  awayGP, isTournament);

  const homeFtr = blend(30, homeInSeason.ftRate, homeGP, isTournament);
  const awayFtr  = blend(30, awayInSeason.ftRate,  awayGP, isTournament);

  const home3pt = blend(33, homeInSeason.threePtPct, homeGP, isTournament);
  const away3pt  = blend(33, awayInSeason.threePtPct,  awayGP, isTournament);

  const home2pt = blend(50, homeInSeason.twoPtPct, homeGP, isTournament);
  const away2pt  = blend(50, awayInSeason.twoPtPct,  awayGP, isTournament);

  const homeBlk = blend(5, homeInSeason.blockPct, homeGP, isTournament);
  const awayBlk  = blend(5, awayInSeason.blockPct,  awayGP, isTournament);

  const homeStl = blend(8, homeInSeason.stealPct, homeGP, isTournament);
  const awayStl  = blend(8, awayInSeason.stealPct,  awayGP, isTournament);

  // ── Strength of schedule ──────────────────────────────────────────────────
  const homeSOS = homeInSeason.sos;
  const awaySOS  = awayInSeason.sos;

  // ── Rest / fatigue ────────────────────────────────────────────────────────
  // We don't have last game date in this simplified version — default to 0 diff
  const restDiff = 0;
  const b2b = 0;

  // ── Home court advantage ──────────────────────────────────────────────────
  const hca = getHomeCourtAdvantage(homeAbbr, homeInSeason.conference, isNeutral);

  // ── Elo diff ──────────────────────────────────────────────────────────────
  const eloDiff = getEloDiff(homeAbbr, awayAbbr);

  // ── Recent form (use adjEM as proxy) ─────────────────────────────────────
  const recentEMDiff = homeAdjEM - awayAdjEM;

  // ── Injury impact (placeholder) ───────────────────────────────────────────
  const injuryImpactDiff = 0;

  return {
    elo_diff: eloDiff,
    adj_em_diff: homeAdjEM - awayAdjEM,
    adj_oe_diff: homeAdjOE - awayAdjOE,
    adj_de_diff: homeAdjDE - awayAdjDE,    // lower = better defense, so negative diff favors home
    adj_tempo_diff: homeAdjTempo - awayAdjTempo,
    pythagorean_diff: homePythagorean - awayPythagorean,
    barthag_diff: homeBarthag - awayBarthag,

    recruiting_composite_diff: homeRecruiting - awayRecruiting,
    returning_minutes_diff: homeReturning - awayReturning,
    transfer_portal_impact_diff: homePortalWAR - awayPortalWAR,
    experience_diff: homeExperience - awayExperience,
    sos_diff: homeSOS - awaySOS,

    efg_pct_diff: homeEfg - awayEfg,
    efg_allowed_diff: homeEfgAllowed - awayEfgAllowed,
    tov_pct_diff: homeTov - awayTov,
    tov_forced_diff: homeTovForced - awayTovForced,
    oreb_pct_diff: homeOreb - awayOreb,
    ft_rate_diff: homeFtr - awayFtr,
    three_pt_pct_diff: home3pt - away3pt,
    two_pt_pct_diff: home2pt - away2pt,
    block_pct_diff: homeBlk - awayBlk,
    steal_pct_diff: homeStl - awayStl,

    team_recent_em_diff: recentEMDiff,
    rest_days_diff: restDiff,
    b2b_flag: b2b,

    is_home: isNeutral ? 0 : 1,
    is_neutral_site: isNeutral ? 1 : 0,
    home_court_advantage: hca,

    injury_impact_diff: injuryImpactDiff,
    vegas_home_prob: 0,

    home_games_played: homeGP,
    away_games_played: awayGP,
    early_season_flag: minGP <= 5 ? 1 : 0,
  };
}

export { earlySeasonLabel };
