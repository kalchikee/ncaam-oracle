#!/usr/bin/env python3
"""
NCAAM March Madness Data Fetcher — last 5 tournaments using ESPN API.
seasontype=3 = postseason/tournament. Output: data/playoff_data.csv

Usage: python python/fetch_playoff_data.py
"""
import sys, json, time, requests
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR  = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache" / "python"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

REG_CSV = DATA_DIR / "historical_games.csv"
OUT_CSV = DATA_DIR / "playoff_data.csv"

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
HEADERS   = {"User-Agent": "NCAAM-Oracle/4.1"}

# Tournament years (season ending year)
TOURNAMENT_YEARS = [2021, 2022, 2023, 2024, 2025]

K_FACTOR   = 20.0
HOME_ADV   = 0.0   # tournaments mostly neutral site
LEAGUE_ELO = 1500.0


def espn_get(url: str) -> dict:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  Failed: {e}"); return {}


def fetch_tournament_games(year: int) -> list:
    """Fetch all March Madness games for a given year."""
    cache = CACHE_DIR / f"ncaam_tournament_{year}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    games = []
    # Tournament spans ~3 weeks in March-April
    start = datetime(year, 3, 14)
    end   = datetime(year, 4, 10)
    current = start

    while current <= end:
        date_str = current.strftime("%Y%m%d")
        url = f"{ESPN_BASE}/scoreboard?dates={date_str}&seasontype=3&limit=50"
        data = espn_get(url)
        for ev in data.get("events", []):
            comps = ev.get("competitions", [{}])[0]
            if not comps.get("competitors"):
                continue
            status = ev.get("status", {}).get("type", {}).get("completed", False)
            if not status:
                current += timedelta(days=1)
                continue
            home = next((c for c in comps["competitors"] if c.get("homeAway") == "home"), None)
            away = next((c for c in comps["competitors"] if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            h_score = int(home.get("score", 0) or 0)
            a_score = int(away.get("score", 0) or 0)
            h_id = home.get("team", {}).get("abbreviation", home.get("team", {}).get("id", ""))
            a_id = away.get("team", {}).get("abbreviation", away.get("team", {}).get("id", ""))
            if not h_id or not a_id or (h_score == 0 and a_score == 0):
                continue
            games.append({
                "game_id":    ev.get("id", ""),
                "game_date":  current.strftime("%Y-%m-%d"),
                "home_team":  str(h_id),
                "away_team":  str(a_id),
                "home_score": h_score,
                "away_score": a_score,
                "neutral":    1,
                "season":     year,
            })
        current += timedelta(days=1)
        time.sleep(0.3)

    cache.write_text(json.dumps(games, indent=2))
    return games


def build_team_stats(reg_df: pd.DataFrame, year: int) -> tuple:
    elo   = defaultdict(lambda: LEAGUE_ELO)
    stats = defaultdict(lambda: {"wins": 0, "games": 0, "pts_for": [], "pts_agst": []})

    s = reg_df[reg_df["season"] == year].sort_values("game_date")
    for _, row in s.iterrows():
        h, a = str(row["home_team"]), str(row["away_team"])
        he = elo[h] + HOME_ADV; ae = elo[a]
        exp = 1 / (1 + 10 ** ((ae - he) / 400))
        act = int(row["home_win"])
        elo[h] += K_FACTOR * (act - exp)
        elo[a] += K_FACTOR * ((1 - act) - (1 - exp))
        stats[h]["wins"]  += act;       stats[a]["wins"]  += (1 - act)
        stats[h]["games"] += 1;         stats[a]["games"] += 1
        h_pts = row.get("home_score", 70); a_pts = row.get("away_score", 70)
        stats[h]["pts_for"].append(h_pts);  stats[h]["pts_agst"].append(a_pts)
        stats[a]["pts_for"].append(a_pts);  stats[a]["pts_agst"].append(h_pts)

        # Use existing feature columns as proxy for quality
    adj_em = {}
    for _, row in s.iterrows():
        h, a = str(row["home_team"]), str(row["away_team"])
        if "adj_em_diff" in row:
            adj_em[h] = row.get("adj_em_diff", 0)

    team_stats = {}
    for team, st in stats.items():
        g = st["games"]
        team_stats[team] = {
            "win_pct": st["wins"] / g if g else 0.5,
            "ppg":     np.mean(st["pts_for"])  if st["pts_for"]  else 70,
            "papg":    np.mean(st["pts_agst"]) if st["pts_agst"] else 70,
        }
    return dict(elo), team_stats


def main():
    print("NCAAM March Madness Data Fetcher")
    print("=" * 40)

    reg_df = pd.read_csv(REG_CSV) if REG_CSV.exists() else pd.DataFrame()
    reg_df["game_date"] = pd.to_datetime(reg_df.get("game_date", pd.Series()), errors="coerce")

    all_rows = []

    for year in TOURNAMENT_YEARS:
        print(f"\nYear {year}")
        elo, stats = build_team_stats(reg_df, year) if not reg_df.empty else ({}, {})
        games = fetch_tournament_games(year)
        print(f"  Fetched {len(games)} tournament games")

        for g in games:
            h, a = g["home_team"], g["away_team"]
            h_elo = elo.get(h, LEAGUE_ELO); a_elo = elo.get(a, LEAGUE_ELO)
            hs  = stats.get(h, {"win_pct": 0.5, "ppg": 70, "papg": 70})
            as_ = stats.get(a, {"win_pct": 0.5, "ppg": 70, "papg": 70})
            label = 1 if g["home_score"] > g["away_score"] else 0

            row = {
                "season":       year,
                "game_id":      g["game_id"],
                "game_date":    g["game_date"],
                "home_team":    h,
                "away_team":    a,
                "home_score":   g["home_score"],
                "away_score":   g["away_score"],
                "home_win":     label,
                "label":        label,
                "is_playoff":   1,
                "is_neutral":   1,
                "elo_diff":     h_elo - a_elo,
                "win_pct_diff": hs["win_pct"] - as_["win_pct"],
                "ppg_diff":     hs["ppg"]  - as_["ppg"],
                "papg_diff":    hs["papg"] - as_["papg"],
            }
            all_rows.append(row)

            exp = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
            elo[h] = h_elo + K_FACTOR * (label - exp)
            elo[a] = a_elo + K_FACTOR * ((1 - label) - (1 - exp))

    if not all_rows:
        print("\nNo data fetched.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} tournament games to {OUT_CSV}")
    print(f"Seasons: {df['season'].unique().tolist()}")


if __name__ == "__main__":
    main()
