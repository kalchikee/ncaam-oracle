#!/usr/bin/env python3
"""
NCAAM Oracle v4.1 — Historical Data Fetcher
Fetches 2018-2025 game data from BartTorvik for model training.
Outputs: data/historical_games.csv
"""

import requests
import pandas as pd
import json
import os
import time
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

BARTTORVIK_BASE = "https://barttorvik.com"

def fetch_barttorvik_games(year: int) -> pd.DataFrame:
    """Fetch all D-I game results for a given season year."""
    url = f"{BARTTORVIK_BASE}/getgames.php?year={year}&csv=1"
    print(f"  Fetching {year} games from BartTorvik...")

    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "NCAAM-Oracle/4.1"})
        resp.raise_for_status()

        # BartTorvik CSV columns: date, home_team, away_team, home_score, away_score,
        # home_adj_o, home_adj_d, away_adj_o, away_adj_d, etc.
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        df["season"] = year
        print(f"  → {len(df)} games")
        return df
    except Exception as e:
        print(f"  Error fetching {year}: {e}")
        return pd.DataFrame()

def fetch_barttorvik_ratings(year: int) -> pd.DataFrame:
    """Fetch season-end T-Rank ratings."""
    url = f"{BARTTORVIK_BASE}/trank.php?year={year}&json=1"
    print(f"  Fetching {year} season ratings...")

    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "NCAAM-Oracle/4.1"})
        resp.raise_for_status()
        data = resp.json()

        rows = data if isinstance(data, list) else data.get("data", [])
        df = pd.DataFrame(rows)
        df["season"] = year
        print(f"  → {len(df)} teams")
        return df
    except Exception as e:
        print(f"  Error fetching {year} ratings: {e}")
        return pd.DataFrame()

def build_training_dataset():
    """Build full historical dataset for model training."""
    print("\n=== NCAAM Oracle — Fetching Historical Data ===\n")

    seasons = range(2019, 2026)  # 2018-19 through 2024-25

    all_games = []
    all_ratings = {}

    for year in seasons:
        print(f"\n[Season {year-1}-{str(year)[-2:]}]")

        games = fetch_barttorvik_games(year)
        ratings = fetch_barttorvik_ratings(year)

        if not games.empty:
            all_games.append(games)
        if not ratings.empty:
            all_ratings[year] = ratings

        time.sleep(1.5)  # Rate limiting

    if not all_games:
        print("No game data fetched! Check BartTorvik availability.")
        return

    # Combine all games
    combined = pd.concat(all_games, ignore_index=True)
    print(f"\nTotal games fetched: {len(combined)}")

    # Save raw games
    combined.to_csv(DATA_DIR / "historical_games.csv", index=False)
    print(f"Saved: {DATA_DIR / 'historical_games.csv'}")

    # Build prior-year data for current season bootstrap
    if 2025 in all_ratings:
        ratings_2025 = all_ratings[2025]
        build_prior_year_json(ratings_2025)

    print("\n=== Historical data fetch complete ===\n")

def build_prior_year_json(ratings_df: pd.DataFrame):
    """Build config/prior_year.json from last season's ratings."""
    print("Building config/prior_year.json from 2024-25 ratings...")

    prior_year = []

    for _, row in ratings_df.iterrows():
        team_abbr = str(row.get("team", "")).upper().replace(" ", "")
        if not team_abbr:
            continue

        # 50% regression to mean for AdjEM (as specified in plan)
        raw_adj_em = float(row.get("adj_o", 100)) - float(row.get("adj_d", 100))
        regressed_adj_em = 0.50 * raw_adj_em + 0.50 * 0  # 50% to D-I mean (0)

        prior_year.append({
            "teamId": 0,
            "teamAbbr": team_abbr,
            "teamName": str(row.get("team", team_abbr)),
            "adjEM": round(regressed_adj_em, 2),
            "adjOE": round(float(row.get("adj_o", 100)), 2),
            "adjDE": round(float(row.get("adj_d", 100)), 2),
            "adjTempo": round(float(row.get("adj_t", 68)), 2),
            "barthag": round(float(row.get("barthag", 0.5)), 4),
            "pythagorean": round(float(row.get("barthag", 0.5)), 4),  # use BARTHAG as proxy
            "recruitingComposite": 0.5,  # placeholder — update from 247Sports
            "returningMinutesPct": 0.55,  # placeholder
            "portalWAR": 0,
            "experienceScore": 2.0,
        })

    with open("config/prior_year.json", "w") as f:
        json.dump(prior_year, f, indent=2)

    print(f"Saved config/prior_year.json ({len(prior_year)} teams)")

if __name__ == "__main__":
    build_training_dataset()
