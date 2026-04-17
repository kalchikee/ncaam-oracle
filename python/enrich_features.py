#!/usr/bin/env python3
"""
NCAAM Oracle — Feature Enrichment
Computes features from game data itself (no external API needed):
  - rest_days_diff: days since each team's last game
  - form_5g_diff: rolling 5-game win% difference
  - efg_pct_diff: estimated eFG% proxy from scoring efficiency
  - tov_pct_diff: estimated turnover rate proxy
  - oreb_pct_diff: estimated offensive rebounding proxy
  - ft_rate_diff: estimated free-throw rate proxy
  - three_pt_pct_diff: estimated 3PT shooting proxy
  - sos_diff: strength of schedule from opponents' win%

Features that need specialized sources (recruiting, transfer portal,
experience, adj_tempo) are left as 0.

Usage: python python/enrich_features.py
"""

import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "historical_games.csv"
BACKUP_PATH = DATA_DIR / "historical_games_backup.csv"


def compute_rest_days(df: pd.DataFrame) -> pd.Series:
    """Compute rest_days_diff = home_rest - away_rest for each game."""
    print("  Computing rest_days_diff...")
    df = df.sort_values("game_date").reset_index(drop=True)

    # Track each team's last game date
    last_game: dict[str, str] = {}
    rest_diff = np.zeros(len(df))

    for i, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        gdate = row["game_date"]

        home_rest = 0.0
        away_rest = 0.0

        if home in last_game:
            try:
                d1 = datetime.strptime(last_game[home], "%Y-%m-%d")
                d2 = datetime.strptime(gdate, "%Y-%m-%d")
                home_rest = (d2 - d1).days
            except (ValueError, TypeError):
                home_rest = 3.0  # default

        if away in last_game:
            try:
                d1 = datetime.strptime(last_game[away], "%Y-%m-%d")
                d2 = datetime.strptime(gdate, "%Y-%m-%d")
                away_rest = (d2 - d1).days
            except (ValueError, TypeError):
                away_rest = 3.0

        # Cap rest at 14 days (longer = offseason or break, not meaningful)
        home_rest = min(home_rest, 14.0)
        away_rest = min(away_rest, 14.0)

        rest_diff[i] = home_rest - away_rest

        last_game[home] = gdate
        last_game[away] = gdate

    return pd.Series(rest_diff, index=df.index)


def compute_form_5g(df: pd.DataFrame) -> pd.Series:
    """Compute form_5g_diff = home 5-game rolling win% - away 5-game rolling win%."""
    print("  Computing form_5g_diff...")
    df = df.sort_values("game_date").reset_index(drop=True)

    # Track each team's last 5 results
    team_results: dict[str, list] = defaultdict(list)
    form_diff = np.zeros(len(df))

    for i, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hw = row["home_win"]

        # Get current form BEFORE this game
        home_form = np.mean(team_results[home][-5:]) if len(team_results[home]) >= 1 else 0.5
        away_form = np.mean(team_results[away][-5:]) if len(team_results[away]) >= 1 else 0.5

        form_diff[i] = home_form - away_form

        # Update results after computing feature
        team_results[home].append(float(hw))
        team_results[away].append(1.0 - float(hw))

    return pd.Series(form_diff, index=df.index)


def compute_scoring_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute four-factors proxies from cumulative scoring data.

    Without per-possession stats (FGA, FTA, TO, OREB counts), we estimate
    shooting efficiency from points-per-game patterns relative to league average.

    For each team, we track:
    - Offensive efficiency: points scored per game (rolling season avg)
    - Defensive efficiency: points allowed per game (rolling season avg)

    Then derive proxies:
    - efg_pct_diff: offensive efficiency differential (higher scoring = better shooting)
    - tov_pct_diff: inverse of scoring consistency (lower variance = fewer turnovers)
    - oreb_pct_diff: margin of victory pattern (close wins suggest 2nd chances)
    - ft_rate_diff: scoring rate above field goals (high PPG vs avg suggests FTs)
    - three_pt_pct_diff: scoring explosiveness (high-scoring games suggest 3PT shooting)
    """
    print("  Computing four-factors proxies from scoring data...")
    df = df.sort_values("game_date").reset_index(drop=True)

    # Track cumulative stats per team per season
    team_stats: dict[str, dict] = {}  # key: (season, team)

    efg_diff = np.zeros(len(df))
    tov_diff = np.zeros(len(df))
    oreb_diff = np.zeros(len(df))
    ft_rate = np.zeros(len(df))
    three_diff = np.zeros(len(df))

    for i, row in df.iterrows():
        season = row["season"]
        home = row["home_team"]
        away = row["away_team"]
        hs = row["home_score"]
        as_ = row["away_score"]

        hkey = (season, home)
        akey = (season, away)

        # Initialize if first game of season
        for key in [hkey, akey]:
            if key not in team_stats:
                team_stats[key] = {
                    "pts_for": [],
                    "pts_against": [],
                    "margins": [],
                    "games": 0,
                }

        h_stats = team_stats[hkey]
        a_stats = team_stats[akey]

        # Compute features BEFORE updating (walk-forward)
        if h_stats["games"] >= 3 and a_stats["games"] >= 3:
            # eFG% proxy: offensive PPG normalized to ~50% scale
            # Average D1 team scores ~72 PPG; eFG% averages ~49%
            h_oe = np.mean(h_stats["pts_for"][-10:]) / 145.0  # normalize to ~0.50
            a_oe = np.mean(a_stats["pts_for"][-10:]) / 145.0
            efg_diff[i] = h_oe - a_oe

            # Turnover proxy: defensive PPG (if you force more TOs, opponent scores less)
            # Lower opponent PPG = better turnover forcing
            h_de = np.mean(h_stats["pts_against"][-10:]) / 145.0
            a_de = np.mean(a_stats["pts_against"][-10:]) / 145.0
            tov_diff[i] = a_de - h_de  # positive = home forces more TOs

            # OREB proxy: margin patterns (consistent positive margins suggest 2nd chances)
            h_margins = h_stats["margins"][-10:]
            a_margins = a_stats["margins"][-10:]
            h_margin_mean = np.mean(h_margins) / 30.0  # normalize
            a_margin_mean = np.mean(a_margins) / 30.0
            oreb_diff[i] = h_margin_mean - a_margin_mean

            # FT rate proxy: points above expected from FG alone
            # Higher PPG relative to opponent PPG suggests FT advantage
            h_ppg = np.mean(h_stats["pts_for"][-10:])
            a_ppg = np.mean(a_stats["pts_for"][-10:])
            h_ft_proxy = (h_ppg - 60.0) / 40.0  # normalize around league avg
            a_ft_proxy = (a_ppg - 60.0) / 40.0
            ft_rate[i] = h_ft_proxy - a_ft_proxy

            # 3PT proxy: scoring variance (3PT heavy teams have higher scoring variance)
            h_std = np.std(h_stats["pts_for"][-10:]) / 15.0 if len(h_stats["pts_for"]) >= 3 else 0.0
            a_std = np.std(a_stats["pts_for"][-10:]) / 15.0 if len(a_stats["pts_for"]) >= 3 else 0.0
            three_diff[i] = h_std - a_std

        # Update stats AFTER computing features
        h_stats["pts_for"].append(hs)
        h_stats["pts_against"].append(as_)
        h_stats["margins"].append(hs - as_)
        h_stats["games"] += 1

        a_stats["pts_for"].append(as_)
        a_stats["pts_against"].append(hs)
        a_stats["margins"].append(as_ - hs)
        a_stats["games"] += 1

    result = pd.DataFrame({
        "efg_pct_diff": efg_diff,
        "tov_pct_diff": tov_diff,
        "oreb_pct_diff": oreb_diff,
        "ft_rate_diff": ft_rate,
        "three_pt_pct_diff": three_diff,
    }, index=df.index)

    return result


def compute_sos(df: pd.DataFrame) -> pd.Series:
    """
    Compute strength of schedule diff.
    SOS = average win% of opponents faced so far this season.
    """
    print("  Computing sos_diff...")
    df = df.sort_values("game_date").reset_index(drop=True)

    # Track per-season team records and opponent lists
    team_record: dict[tuple, dict] = {}  # (season, team) -> {wins, losses}
    team_opponents: dict[tuple, list] = {}  # (season, team) -> [opponent_keys]

    sos_diff = np.zeros(len(df))

    for i, row in df.iterrows():
        season = row["season"]
        home = row["home_team"]
        away = row["away_team"]
        hw = row["home_win"]

        hkey = (season, home)
        akey = (season, away)

        for key in [hkey, akey]:
            if key not in team_record:
                team_record[key] = {"wins": 0, "losses": 0}
                team_opponents[key] = []

        # Compute SOS BEFORE updating
        h_opps = team_opponents[hkey]
        a_opps = team_opponents[akey]

        if len(h_opps) >= 3 and len(a_opps) >= 3:
            h_sos = np.mean([
                team_record[opp]["wins"] / max(1, team_record[opp]["wins"] + team_record[opp]["losses"])
                for opp in h_opps if opp in team_record
            ]) if h_opps else 0.5

            a_sos = np.mean([
                team_record[opp]["wins"] / max(1, team_record[opp]["wins"] + team_record[opp]["losses"])
                for opp in a_opps if opp in team_record
            ]) if a_opps else 0.5

            sos_diff[i] = h_sos - a_sos

        # Update records
        team_record[hkey]["wins"] += int(hw)
        team_record[hkey]["losses"] += int(1 - hw)
        team_record[akey]["wins"] += int(1 - hw)
        team_record[akey]["losses"] += int(hw)

        team_opponents[hkey].append(akey)
        team_opponents[akey].append(hkey)

    return pd.Series(sos_diff, index=df.index)


def compute_efg_allowed(df: pd.DataFrame) -> pd.Series:
    """
    Compute efg_allowed_diff: defensive eFG% proxy.
    Lower opponent scoring = better defensive eFG%.
    """
    print("  Computing efg_allowed_diff...")
    df = df.sort_values("game_date").reset_index(drop=True)

    team_def: dict[tuple, list] = {}  # (season, team) -> [pts_allowed]
    efg_allowed = np.zeros(len(df))

    for i, row in df.iterrows():
        season = row["season"]
        home = row["home_team"]
        away = row["away_team"]

        hkey = (season, home)
        akey = (season, away)

        for key in [hkey, akey]:
            if key not in team_def:
                team_def[key] = []

        if len(team_def[hkey]) >= 3 and len(team_def[akey]) >= 3:
            h_def = np.mean(team_def[hkey][-10:]) / 145.0
            a_def = np.mean(team_def[akey][-10:]) / 145.0
            # Lower allowed = better defense, so we want home_allowed < away_allowed = positive diff for home
            efg_allowed[i] = a_def - h_def  # positive = home has better defense

        team_def[hkey].append(row["away_score"])
        team_def[akey].append(row["home_score"])

    return pd.Series(efg_allowed, index=df.index)


def compute_three_pt_rate(df: pd.DataFrame) -> pd.Series:
    """
    three_pt_rate_diff: proxy for how much a team relies on 3PT shooting.
    High variance + high scoring suggests 3PT reliance.
    """
    print("  Computing three_pt_rate_diff...")
    df = df.sort_values("game_date").reset_index(drop=True)

    team_scores: dict[tuple, list] = {}
    rate_diff = np.zeros(len(df))

    for i, row in df.iterrows():
        season = row["season"]
        home = row["home_team"]
        away = row["away_team"]

        hkey = (season, home)
        akey = (season, away)

        for key in [hkey, akey]:
            if key not in team_scores:
                team_scores[key] = []

        if len(team_scores[hkey]) >= 5 and len(team_scores[akey]) >= 5:
            h_scores = team_scores[hkey][-10:]
            a_scores = team_scores[akey][-10:]
            # Coefficient of variation as 3PT reliance proxy
            h_cv = np.std(h_scores) / max(np.mean(h_scores), 1.0)
            a_cv = np.std(a_scores) / max(np.mean(a_scores), 1.0)
            rate_diff[i] = h_cv - a_cv

        team_scores[hkey].append(row["home_score"])
        team_scores[akey].append(row["away_score"])

    return pd.Series(rate_diff, index=df.index)


def compute_tov_forced(df: pd.DataFrame) -> pd.Series:
    """
    tov_forced_diff: proxy for defensive turnover forcing.
    Teams that hold opponents to low scores relative to their avg likely force more TOs.
    """
    print("  Computing tov_forced_diff...")
    df = df.sort_values("game_date").reset_index(drop=True)

    # Track each team's offensive average and each team's defensive suppression
    team_opp_scoring: dict[tuple, list] = {}  # points opponents score against this team
    team_own_scoring: dict[tuple, list] = {}   # points this team scores
    tov_forced = np.zeros(len(df))

    for i, row in df.iterrows():
        season = row["season"]
        home = row["home_team"]
        away = row["away_team"]

        hkey = (season, home)
        akey = (season, away)

        for key in [hkey, akey]:
            if key not in team_opp_scoring:
                team_opp_scoring[key] = []
                team_own_scoring[key] = []

        if len(team_opp_scoring[hkey]) >= 3 and len(team_opp_scoring[akey]) >= 3:
            # How much does this team suppress opponent scoring below their average?
            h_opp_avg = np.mean(team_opp_scoring[hkey][-10:])
            a_opp_avg = np.mean(team_opp_scoring[akey][-10:])
            # Normalize: lower opponent scoring = more turnovers forced
            h_suppression = (72.0 - h_opp_avg) / 20.0  # 72 is approx D1 avg
            a_suppression = (72.0 - a_opp_avg) / 20.0
            tov_forced[i] = h_suppression - a_suppression

        team_opp_scoring[hkey].append(row["away_score"])
        team_opp_scoring[akey].append(row["home_score"])
        team_own_scoring[hkey].append(row["home_score"])
        team_own_scoring[akey].append(row["away_score"])

    return pd.Series(tov_forced, index=df.index)


def main():
    print("=== NCAAM Oracle — Feature Enrichment ===\n")

    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run fetch_historical.py first.")
        return

    # Load
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} games from {CSV_PATH}")
    print(f"Columns: {df.columns.tolist()}")

    # Backup
    shutil.copy2(CSV_PATH, BACKUP_PATH)
    print(f"Backup saved to {BACKUP_PATH}")

    # Sort by date for walk-forward computation
    df = df.sort_values("game_date").reset_index(drop=True)

    # 1. Rest days
    df["rest_days_diff"] = compute_rest_days(df)

    # 2. Rolling 5-game form
    df["form_5g_diff"] = compute_form_5g(df)

    # 3. Four-factors proxies from scoring data
    scoring_features = compute_scoring_efficiency_features(df)
    df["efg_pct_diff"] = scoring_features["efg_pct_diff"]
    df["tov_pct_diff"] = scoring_features["tov_pct_diff"]
    df["oreb_pct_diff"] = scoring_features["oreb_pct_diff"]
    df["ft_rate_diff"] = scoring_features["ft_rate_diff"]
    df["three_pt_pct_diff"] = scoring_features["three_pt_pct_diff"]

    # 4. eFG allowed (defensive)
    df["efg_allowed_diff"] = compute_efg_allowed(df)

    # 5. Three-point rate
    df["three_pt_rate_diff"] = compute_three_pt_rate(df)

    # 6. Turnover forced
    df["tov_forced_diff"] = compute_tov_forced(df)

    # 7. Strength of schedule
    df["sos_diff"] = compute_sos(df)

    # Save
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved enriched data to {CSV_PATH}")

    # Report
    print("\n--- Enrichment Summary ---")
    enriched_cols = [
        "rest_days_diff", "form_5g_diff", "efg_pct_diff", "efg_allowed_diff",
        "tov_pct_diff", "tov_forced_diff", "oreb_pct_diff", "ft_rate_diff",
        "three_pt_pct_diff", "three_pt_rate_diff", "sos_diff",
    ]
    for col in enriched_cols:
        nonzero = (df[col] != 0).sum()
        print(f"  {col:30s}  non-zero: {nonzero:6d}/{len(df)}  mean={df[col].mean():.4f}  std={df[col].std():.4f}")

    still_zero = [
        "adj_tempo_diff", "recruiting_composite_diff", "returning_minutes_diff",
        "transfer_portal_impact_diff", "experience_diff", "vegas_home_prob", "mc_win_pct",
    ]
    print(f"\n  Still zero (need specialized sources): {still_zero}")
    print("\nDone! Run: python python/train_model.py")


if __name__ == "__main__":
    main()
