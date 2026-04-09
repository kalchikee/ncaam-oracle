// NCAAM Oracle v4.1 — Core Type Definitions

// ─── CBB Team & Game types ───────────────────────────────────────────────────

export interface CBBTeam {
  teamId: number;           // ESPN team ID
  teamAbbr: string;         // Short name / abbr used internally
  teamName: string;         // Full display name
  conference: string;
  // Current-season stats (in-season)
  gamesPlayed: number;
  adjEM: number;            // Adjusted Efficiency Margin (KenPom / BartTorvik)
  adjOE: number;            // Adjusted Offensive Efficiency
  adjDE: number;            // Adjusted Defensive Efficiency
  adjTempo: number;         // Adjusted Tempo (possessions per 40 min)
  barthag: number;          // BartTorvik BARTHAG (power rating, 0-1)
  wab: number;              // Wins Above Bubble
  efgPct: number;           // Effective FG%
  efgAllowed: number;       // Opponent EFG%
  tovPct: number;           // Turnover %
  tovForced: number;        // Opponent TOV %
  orebPct: number;          // Offensive Rebound %
  drebPct: number;          // Defensive Rebound %
  ftRate: number;           // Free Throw Rate (FTA/FGA)
  ftRateAllowed: number;    // Opponent FTR
  threePtPct: number;       // 3-point %
  twoPtPct: number;         // 2-point %
  blockPct: number;         // Block %
  stealPct: number;         // Steal %
  winPct: number;           // Win percentage
  // Strength of schedule
  sos: number;
  ncaaNET?: number;         // NCAA NET ranking
}

export interface CBBGame {
  gameId: string;
  gameDate: string;         // YYYY-MM-DD
  gameTime: string;         // ISO datetime
  status: string;           // 'Scheduled' | 'Live' | 'Final'
  homeTeam: CBBGameTeam;
  awayTeam: CBBGameTeam;
  arena: string;
  isNeutralSite: boolean;
  isTournamentGame: boolean;
  tournamentRound?: string;
}

export interface CBBGameTeam {
  teamId: number;
  teamAbbr: string;
  teamName: string;
  score?: number;
}

// ─── Prior-year bootstrap data ────────────────────────────────────────────────

export interface PriorYearData {
  teamId: number;
  teamAbbr: string;
  teamName: string;
  adjEM: number;            // Prior-year final AdjEM (with 50% regression applied)
  adjOE: number;
  adjDE: number;
  adjTempo: number;
  barthag: number;
  pythagorean: number;      // Pythagorean win%
  recruitingComposite: number;  // 247Sports composite (normalized)
  returningMinutesPct: number;  // % of minutes returning
  portalWAR: number;        // Transfer portal wins above replacement
  experienceScore: number;  // Avg experience (years)
}

// ─── Feature vector ───────────────────────────────────────────────────────────

export interface FeatureVector {
  // Team strength (available from game 1 via prior-year bootstrap)
  elo_diff: number;
  adj_em_diff: number;
  adj_oe_diff: number;
  adj_de_diff: number;
  adj_tempo_diff: number;
  pythagorean_diff: number;
  barthag_diff: number;

  // Recruiting & roster (prior-year bootstrap)
  recruiting_composite_diff: number;
  returning_minutes_diff: number;
  transfer_portal_impact_diff: number;
  experience_diff: number;
  sos_diff: number;

  // In-season shooting efficiency (activates after early season)
  efg_pct_diff: number;
  efg_allowed_diff: number;
  tov_pct_diff: number;
  tov_forced_diff: number;
  oreb_pct_diff: number;
  ft_rate_diff: number;
  three_pt_pct_diff: number;
  two_pt_pct_diff: number;
  block_pct_diff: number;
  steal_pct_diff: number;

  // Form
  team_recent_em_diff: number;

  // Fatigue & scheduling
  rest_days_diff: number;
  b2b_flag: number;

  // Venue
  is_home: number;
  is_neutral_site: number;
  home_court_advantage: number;

  // Injury impact
  injury_impact_diff: number;

  // Vegas (filled at prediction time)
  vegas_home_prob: number;

  // Bootstrap metadata (not a model feature, used for labeling)
  home_games_played: number;
  away_games_played: number;
  early_season_flag: number;
}

// ─── Model outputs ────────────────────────────────────────────────────────────

export interface MonteCarloResult {
  win_probability: number;
  away_win_probability: number;
  spread: number;
  total_points: number;
  most_likely_score: [number, number];
  home_exp_pts: number;
  away_exp_pts: number;
  simulations: number;
}

export interface Prediction {
  game_date: string;
  game_id: string;
  home_team: string;
  away_team: string;
  arena: string;
  is_neutral_site: boolean;
  is_tournament: boolean;
  feature_vector: FeatureVector;
  mc_win_pct: number;
  calibrated_prob: number;
  vegas_prob?: number;
  edge?: number;
  model_version: string;
  home_exp_pts: number;
  away_exp_pts: number;
  total_points: number;
  spread: number;
  most_likely_score: string;
  early_season_label?: string;
  actual_winner?: string;
  correct?: boolean;
  created_at: string;
}

export interface EloRating {
  teamAbbr: string;
  rating: number;
  updatedAt: string;
}

export interface SeasonAccuracy {
  season: string;
  date: string;
  total_correct: number;
  total_picks: number;
  accuracy_pct: number;
  high_conv_correct: number;
  high_conv_total: number;
  brier: number;
  ats_wins: number;
  ats_losses: number;
  tournament_correct: number;
  tournament_total: number;
}

export interface WeeklyAccuracy {
  season: string;
  week_num: number;
  week_start: string;
  correct: number;
  total: number;
  accuracy_pct: number;
  brier: number;
}

export interface TournamentSim {
  season: string;
  team_abbr: string;
  team_name: string;
  seed: number;
  region: string;
  r64_prob: number;
  r32_prob: number;
  s16_prob: number;
  e8_prob: number;
  f4_prob: number;
  champ_prob: number;
}

export interface PipelineOptions {
  date?: string;
  forceRefresh?: boolean;
  verbose?: boolean;
  tournamentMode?: boolean;
}

// ─── Confidence tiers ─────────────────────────────────────────────────────────

export type ConfidenceTier = 'coin_flip' | 'lean' | 'strong' | 'high_conviction' | 'extreme';

export interface EdgeResult {
  modelProb: number;
  vegasProb: number;
  edge: number;
  edgeCategory: 'none' | 'small' | 'meaningful' | 'large';
}
