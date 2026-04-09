// NCAAM Oracle v4.1 — SQLite Database Layer (sql.js — pure JS, no native build)

import initSqlJs, { type Database as SqlJsDatabase } from 'sql.js';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { resolve } from 'path';
import type {
  Prediction, EloRating, SeasonAccuracy, WeeklyAccuracy, TournamentSim,
} from '../types.js';

const DB_PATH = resolve(
  process.env.DB_PATH ?? resolve('data/oracle.sqlite')
);

mkdirSync(resolve('data'), { recursive: true });

let _db: SqlJsDatabase | null = null;
let _SQL: Awaited<ReturnType<typeof initSqlJs>> | null = null;

// ─── Initialization ───────────────────────────────────────────────────────────

export async function initDb(): Promise<SqlJsDatabase> {
  if (_db) return _db;

  _SQL = await initSqlJs();

  if (existsSync(DB_PATH)) {
    const fileBuffer = readFileSync(DB_PATH);
    _db = new _SQL.Database(fileBuffer);
  } else {
    _db = new _SQL.Database();
  }

  initializeSchema(_db);
  persistDb();
  return _db;
}

export function getDb(): SqlJsDatabase {
  if (!_db) throw new Error('Database not initialized. Call initDb() first.');
  return _db;
}

export function persistDb(): void {
  if (!_db) return;
  const data = _db.export();
  writeFileSync(DB_PATH, Buffer.from(data));
}

export function closeDb(): void {
  persistDb();
  _db?.close();
  _db = null;
}

// ─── Query helpers ────────────────────────────────────────────────────────────

function run(sql: string, params: (string | number | null | undefined)[] = []): void {
  const db = getDb();
  const stmt = db.prepare(sql);
  stmt.run(params.map(p => p === undefined ? null : p));
  stmt.free();
  persistDb();
}

function queryAll<T = Record<string, unknown>>(
  sql: string,
  params: (string | number | null)[] = []
): T[] {
  const db = getDb();
  const stmt = db.prepare(sql);
  stmt.bind(params);
  const results: T[] = [];
  while (stmt.step()) results.push(stmt.getAsObject() as T);
  stmt.free();
  return results;
}

function queryOne<T = Record<string, unknown>>(
  sql: string,
  params: (string | number | null)[] = []
): T | undefined {
  return queryAll<T>(sql, params)[0];
}

// ─── Schema ───────────────────────────────────────────────────────────────────

function initializeSchema(db: SqlJsDatabase): void {
  db.run(`
    CREATE TABLE IF NOT EXISTS elo_ratings (
      team_abbr TEXT PRIMARY KEY,
      rating REAL NOT NULL DEFAULT 1500,
      updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS predictions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      game_id TEXT NOT NULL,
      game_date TEXT NOT NULL,
      home_team TEXT NOT NULL,
      away_team TEXT NOT NULL,
      arena TEXT NOT NULL DEFAULT '',
      is_neutral_site INTEGER NOT NULL DEFAULT 0,
      is_tournament INTEGER NOT NULL DEFAULT 0,
      feature_vector TEXT NOT NULL,
      home_exp_pts REAL NOT NULL DEFAULT 0,
      away_exp_pts REAL NOT NULL DEFAULT 0,
      total_points REAL NOT NULL DEFAULT 0,
      spread REAL NOT NULL DEFAULT 0,
      most_likely_score TEXT NOT NULL DEFAULT '',
      mc_win_pct REAL NOT NULL,
      calibrated_prob REAL NOT NULL,
      vegas_prob REAL,
      edge REAL,
      early_season_label TEXT,
      model_version TEXT NOT NULL DEFAULT '4.1.0',
      actual_winner TEXT,
      correct INTEGER,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      UNIQUE(game_id, game_date)
    );

    CREATE TABLE IF NOT EXISTS season_accuracy (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      season TEXT NOT NULL,
      date TEXT NOT NULL,
      total_correct INTEGER NOT NULL DEFAULT 0,
      total_picks INTEGER NOT NULL DEFAULT 0,
      accuracy_pct REAL NOT NULL DEFAULT 0,
      high_conv_correct INTEGER NOT NULL DEFAULT 0,
      high_conv_total INTEGER NOT NULL DEFAULT 0,
      brier REAL NOT NULL DEFAULT 0,
      ats_wins INTEGER NOT NULL DEFAULT 0,
      ats_losses INTEGER NOT NULL DEFAULT 0,
      tournament_correct INTEGER NOT NULL DEFAULT 0,
      tournament_total INTEGER NOT NULL DEFAULT 0,
      UNIQUE(season, date)
    );

    CREATE TABLE IF NOT EXISTS weekly_accuracy (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      season TEXT NOT NULL,
      week_num INTEGER NOT NULL,
      week_start TEXT NOT NULL,
      correct INTEGER NOT NULL DEFAULT 0,
      total INTEGER NOT NULL DEFAULT 0,
      accuracy_pct REAL NOT NULL DEFAULT 0,
      brier REAL NOT NULL DEFAULT 0,
      UNIQUE(season, week_num)
    );

    CREATE TABLE IF NOT EXISTS tournament_sims (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      season TEXT NOT NULL,
      team_abbr TEXT NOT NULL,
      team_name TEXT NOT NULL,
      seed INTEGER NOT NULL DEFAULT 16,
      region TEXT NOT NULL DEFAULT '',
      r64_prob REAL NOT NULL DEFAULT 0,
      r32_prob REAL NOT NULL DEFAULT 0,
      s16_prob REAL NOT NULL DEFAULT 0,
      e8_prob REAL NOT NULL DEFAULT 0,
      f4_prob REAL NOT NULL DEFAULT 0,
      champ_prob REAL NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      UNIQUE(season, team_abbr)
    );
  `);
}

// ─── Elo CRUD ─────────────────────────────────────────────────────────────────

export function getElo(teamAbbr: string): number {
  const row = queryOne<{ rating: number }>(
    'SELECT rating FROM elo_ratings WHERE team_abbr = ?',
    [teamAbbr]
  );
  return row?.rating ?? 1500;
}

export function upsertElo(elo: EloRating): void {
  run(
    `INSERT INTO elo_ratings (team_abbr, rating, updated_at)
     VALUES (?, ?, ?)
     ON CONFLICT(team_abbr) DO UPDATE SET rating = excluded.rating, updated_at = excluded.updated_at`,
    [elo.teamAbbr, elo.rating, elo.updatedAt]
  );
}

export function getAllElos(): EloRating[] {
  return queryAll<EloRating>(
    'SELECT team_abbr as teamAbbr, rating, updated_at as updatedAt FROM elo_ratings ORDER BY rating DESC'
  );
}

// ─── Predictions CRUD ─────────────────────────────────────────────────────────

export function upsertPrediction(p: Prediction): void {
  run(
    `INSERT INTO predictions (
      game_id, game_date, home_team, away_team, arena,
      is_neutral_site, is_tournament, feature_vector,
      home_exp_pts, away_exp_pts, total_points, spread, most_likely_score,
      mc_win_pct, calibrated_prob, vegas_prob, edge, early_season_label,
      model_version, created_at
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ON CONFLICT(game_id, game_date) DO UPDATE SET
      calibrated_prob = excluded.calibrated_prob,
      mc_win_pct = excluded.mc_win_pct,
      vegas_prob = excluded.vegas_prob,
      edge = excluded.edge,
      home_exp_pts = excluded.home_exp_pts,
      away_exp_pts = excluded.away_exp_pts,
      total_points = excluded.total_points,
      spread = excluded.spread`,
    [
      p.game_id, p.game_date, p.home_team, p.away_team, p.arena,
      p.is_neutral_site ? 1 : 0,
      p.is_tournament ? 1 : 0,
      JSON.stringify(p.feature_vector),
      p.home_exp_pts, p.away_exp_pts, p.total_points, p.spread, p.most_likely_score,
      p.mc_win_pct, p.calibrated_prob,
      p.vegas_prob ?? null, p.edge ?? null,
      p.early_season_label ?? null,
      p.model_version, p.created_at,
    ]
  );
}

export function getPredictionsByDate(date: string): Prediction[] {
  const rows = queryAll<Record<string, unknown>>(
    'SELECT * FROM predictions WHERE game_date = ? ORDER BY calibrated_prob DESC',
    [date]
  );
  return rows.map(rowToPrediction);
}

export function markPredictionResult(
  gameId: string,
  date: string,
  homeScore: number,
  awayScore: number,
  homeTeam: string,
  awayTeam: string
): void {
  const actualWinner = homeScore > awayScore ? homeTeam : awayTeam;

  // Get the prediction to determine if we were correct
  const pred = queryOne<{ calibrated_prob: number; home_team: string }>(
    'SELECT calibrated_prob, home_team FROM predictions WHERE game_id = ? AND game_date = ?',
    [gameId, date]
  );

  if (!pred) return;

  const pickedHome = pred.calibrated_prob >= 0.5;
  const homeWon = homeScore > awayScore;
  const correct = pickedHome === homeWon ? 1 : 0;

  run(
    `UPDATE predictions SET actual_winner = ?, correct = ? WHERE game_id = ? AND game_date = ?`,
    [actualWinner, correct, gameId, date]
  );
}

export function getUnscoredPredictions(date: string): Array<{ game_id: string; home_team: string; away_team: string; calibrated_prob: number; vegas_prob: number | null }> {
  return queryAll(
    `SELECT game_id, home_team, away_team, calibrated_prob, vegas_prob
     FROM predictions WHERE game_date = ? AND correct IS NULL`,
    [date]
  );
}

// ─── Season accuracy ──────────────────────────────────────────────────────────

export function upsertSeasonAccuracy(acc: SeasonAccuracy): void {
  run(
    `INSERT INTO season_accuracy
      (season, date, total_correct, total_picks, accuracy_pct, high_conv_correct, high_conv_total,
       brier, ats_wins, ats_losses, tournament_correct, tournament_total)
     VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
     ON CONFLICT(season, date) DO UPDATE SET
      total_correct = excluded.total_correct,
      total_picks = excluded.total_picks,
      accuracy_pct = excluded.accuracy_pct,
      high_conv_correct = excluded.high_conv_correct,
      high_conv_total = excluded.high_conv_total,
      brier = excluded.brier,
      ats_wins = excluded.ats_wins,
      ats_losses = excluded.ats_losses,
      tournament_correct = excluded.tournament_correct,
      tournament_total = excluded.tournament_total`,
    [
      acc.season, acc.date, acc.total_correct, acc.total_picks, acc.accuracy_pct,
      acc.high_conv_correct, acc.high_conv_total, acc.brier,
      acc.ats_wins, acc.ats_losses, acc.tournament_correct, acc.tournament_total,
    ]
  );
}

export function getLatestSeasonAccuracy(season: string): SeasonAccuracy | null {
  const row = queryOne<Record<string, unknown>>(
    `SELECT * FROM season_accuracy WHERE season = ? ORDER BY date DESC LIMIT 1`,
    [season]
  );
  if (!row) return null;
  return row as unknown as SeasonAccuracy;
}

export function getSeasonRecord(season: string): { correct: number; total: number; highConvCorrect: number; highConvTotal: number; atsWins: number; atsLosses: number; brier: number } {
  // Sum from all predictions with results
  const rows = queryAll<{ calibrated_prob: number; correct: number; vegas_prob: number | null; is_tournament: number }>(
    `SELECT calibrated_prob, correct, vegas_prob, is_tournament FROM predictions
     WHERE game_date LIKE ? AND correct IS NOT NULL`,
    [`${season.split('-')[0]}-%`]
  );

  let correct = 0, total = 0;
  let hcCorrect = 0, hcTotal = 0;
  let atsWins = 0, atsLosses = 0;
  let brierSum = 0;

  for (const r of rows) {
    total++;
    if (r.correct) correct++;

    const p = Math.max(r.calibrated_prob, 1 - r.calibrated_prob);
    brierSum += Math.pow(r.calibrated_prob - (r.correct ? 1 : 0), 2);

    if (p >= 0.70) {
      hcTotal++;
      if (r.correct) hcCorrect++;
    }

    if (r.vegas_prob !== null && r.vegas_prob !== undefined) {
      const modelFavorsHome = r.calibrated_prob >= 0.5;
      const vegasFavorsHome = r.vegas_prob >= 0.5;
      if (modelFavorsHome !== vegasFavorsHome) {
        if (r.correct) atsWins++;
        else atsLosses++;
      }
    }
  }

  return {
    correct,
    total,
    highConvCorrect: hcCorrect,
    highConvTotal: hcTotal,
    atsWins,
    atsLosses,
    brier: total > 0 ? brierSum / total : 0,
  };
}

export function getWeeklyRecords(season: string, weeksBack = 8): Array<{ weekStart: string; correct: number; total: number; accuracy: number }> {
  // Group predictions by ISO week
  const rows = queryAll<{ game_date: string; correct: number }>(
    `SELECT game_date, correct FROM predictions
     WHERE game_date LIKE ? AND correct IS NOT NULL
     ORDER BY game_date`,
    [`${season.split('-')[0]}-%`]
  );

  const weekMap = new Map<string, { correct: number; total: number }>();

  for (const r of rows) {
    const d = new Date(r.game_date);
    const day = d.getDay();
    const monday = new Date(d);
    monday.setDate(d.getDate() - ((day + 6) % 7));
    const weekKey = monday.toISOString().split('T')[0];

    const entry = weekMap.get(weekKey) ?? { correct: 0, total: 0 };
    entry.total++;
    if (r.correct) entry.correct++;
    weekMap.set(weekKey, entry);
  }

  return Array.from(weekMap.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .slice(-weeksBack)
    .map(([weekStart, { correct, total }]) => ({
      weekStart,
      correct,
      total,
      accuracy: total > 0 ? correct / total : 0,
    }));
}

// ─── Tournament sims ──────────────────────────────────────────────────────────

export function upsertTournamentSim(sim: TournamentSim): void {
  run(
    `INSERT INTO tournament_sims
      (season, team_abbr, team_name, seed, region, r64_prob, r32_prob, s16_prob, e8_prob, f4_prob, champ_prob)
     VALUES (?,?,?,?,?,?,?,?,?,?,?)
     ON CONFLICT(season, team_abbr) DO UPDATE SET
      r64_prob = excluded.r64_prob, r32_prob = excluded.r32_prob,
      s16_prob = excluded.s16_prob, e8_prob = excluded.e8_prob,
      f4_prob = excluded.f4_prob, champ_prob = excluded.champ_prob`,
    [
      sim.season, sim.team_abbr, sim.team_name, sim.seed, sim.region,
      sim.r64_prob, sim.r32_prob, sim.s16_prob, sim.e8_prob, sim.f4_prob, sim.champ_prob,
    ]
  );
}

export function getTournamentSims(season: string): TournamentSim[] {
  return queryAll<TournamentSim>(
    `SELECT * FROM tournament_sims WHERE season = ? ORDER BY champ_prob DESC`,
    [season]
  );
}

// ─── Row mapping ──────────────────────────────────────────────────────────────

function rowToPrediction(row: Record<string, unknown>): Prediction {
  return {
    game_date: row['game_date'] as string,
    game_id: row['game_id'] as string,
    home_team: row['home_team'] as string,
    away_team: row['away_team'] as string,
    arena: row['arena'] as string,
    is_neutral_site: Boolean(row['is_neutral_site']),
    is_tournament: Boolean(row['is_tournament']),
    feature_vector: JSON.parse(row['feature_vector'] as string),
    mc_win_pct: row['mc_win_pct'] as number,
    calibrated_prob: row['calibrated_prob'] as number,
    vegas_prob: row['vegas_prob'] as number | undefined,
    edge: row['edge'] as number | undefined,
    model_version: row['model_version'] as string,
    home_exp_pts: row['home_exp_pts'] as number,
    away_exp_pts: row['away_exp_pts'] as number,
    total_points: row['total_points'] as number,
    spread: row['spread'] as number,
    most_likely_score: row['most_likely_score'] as string,
    early_season_label: row['early_season_label'] as string | undefined,
    actual_winner: row['actual_winner'] as string | undefined,
    correct: row['correct'] !== null ? Boolean(row['correct']) : undefined,
    created_at: row['created_at'] as string,
  };
}
