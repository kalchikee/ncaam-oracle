#!/usr/bin/env python3
"""
NCAAM Oracle v4.1 — Live Predictions
Fetches today's/upcoming men's college basketball games from ESPN,
computes features using season stats + trained Elo, runs the logistic
regression meta-model, and prints predictions.

Usage:
  python python/predict.py            # today's games
  python python/predict.py --date 20260412
"""

import argparse
import json
import math
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
MODEL_DIR   = ROOT / "model"
DATA_DIR    = ROOT / "data"
HIST_CSV    = DATA_DIR / "historical_games.csv"

ESPN_BASE   = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
HEADERS     = {"User-Agent": "NCAAM-Oracle/4.1"}

CBB_TEMPO   = 68.0
INITIAL_ELO = 1500.0
K_FACTOR    = 20.0
HOME_ADV    = 100.0


# ── ESPN helpers ───────────────────────────────────────────────────────────────

def fetch_json(url: str) -> dict | None:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    return None


def fetch_games(date_str: str) -> list[dict]:
    url = f"{ESPN_BASE}/scoreboard?dates={date_str}&groups=50&limit=200"
    data = fetch_json(url)
    if not data:
        return []
    games = []
    for event in data.get("events", []):
        status = event.get("status", {}).get("type", {}).get("name", "")
        comp   = (event.get("competitions") or [{}])[0]
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        games.append({
            "event_id":   event.get("id", ""),
            "event_name": event.get("name", ""),
            "status":     status,
            "date":       event.get("date", "")[:10],
            "neutral":    int(comp.get("neutralSite", False)),
            "home_id":    home.get("team", {}).get("id", ""),
            "home_abbr":  home.get("team", {}).get("abbreviation", "").upper(),
            "home_name":  home.get("team", {}).get("displayName", ""),
            "away_id":    away.get("team", {}).get("id", ""),
            "away_abbr":  away.get("team", {}).get("abbreviation", "").upper(),
            "away_name":  away.get("team", {}).get("displayName", ""),
        })
    return games


def fetch_team_stats(team_id: str) -> dict:
    """Fetch season record stats from ESPN for a team."""
    url = f"{ESPN_BASE}/teams/{team_id}?enable=record,stats"
    data = fetch_json(url)
    if not data:
        return {}
    items = data.get("team", {}).get("record", {}).get("items", [])
    total = next((i for i in items if i.get("type") == "total"), {})
    stats = {s["name"]: s["value"] for s in total.get("stats", [])}
    return stats


# ── Elo reconstruction ─────────────────────────────────────────────────────────

def build_elo_from_history() -> dict[str, float]:
    """Replay historical games to get current Elo ratings."""
    if not HIST_CSV.exists():
        return {}
    df = pd.read_csv(HIST_CSV, usecols=["game_date", "season", "home_team", "away_team",
                                         "elo_diff", "home_win", "home_score", "away_score"])
    df = df.sort_values("game_date")

    elo: dict[str, float] = {}
    last_season = None

    for _, row in df.iterrows():
        season = row["season"]
        if last_season is not None and season != last_season:
            for tid in elo:
                elo[tid] = 0.70 * elo[tid] + 0.30 * INITIAL_ELO
        last_season = season

        h, a    = row["home_team"], row["away_team"]
        hw      = int(row["home_win"])
        margin  = abs(int(row["home_score"]) - int(row["away_score"]))
        rh      = elo.get(h, INITIAL_ELO)
        ra      = elo.get(a, INITIAL_ELO)
        neutral = 0  # not stored in CSV; assume non-neutral for Elo update
        ha      = 0 if neutral else HOME_ADV
        exp_h   = 1.0 / (1.0 + 10 ** ((ra - (rh + ha)) / 400.0))
        mov     = math.log(1 + min(margin, 35))
        k       = K_FACTOR * mov
        delta   = k * (hw - exp_h)
        elo[h]  = rh + delta
        elo[a]  = ra - delta

    return elo


# ── Model loading ──────────────────────────────────────────────────────────────

def is_tournament_season(date_str: str) -> bool:
    """NCAA Tournament runs mid-March through early April."""
    try:
        d = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        return False
    return (d.month == 3 and d.day >= 14) or (d.month == 4 and d.day <= 10)


def load_model() -> dict | None:
    try:
        # Tournament season: use playoff-specific model if available
        if is_tournament_season(datetime.now().strftime("%Y%m%d")):
            po = MODEL_DIR / "playoff_coefficients.json"
            ps = MODEL_DIR / "playoff_scaler.json"
            pm = MODEL_DIR / "playoff_metadata.json"
            if po.exists() and ps.exists() and pm.exists():
                po_data = json.loads(po.read_text())
                # Convert list format to dict format expected by predict_proba
                coeff_dict = dict(zip(po_data["feature_names"], po_data["coefficients"]))
                coeff_dict["intercept"] = po_data["intercept"]
                meta = json.loads(pm.read_text())
                meta["feature_names"] = po_data["feature_names"]
                scaler = json.loads(ps.read_text())
                print("  [TOURNAMENT MODE] Using March Madness playoff model")
                return {"coeff": coeff_dict, "scaler": scaler,
                        "calib": {"bins": [], "calibrated": []}, "meta": meta}
        coeff = json.loads((MODEL_DIR / "coefficients.json").read_text())
        scaler = json.loads((MODEL_DIR / "scaler.json").read_text())
        calib  = json.loads((MODEL_DIR / "calibration.json").read_text())
        meta   = json.loads((MODEL_DIR / "metadata.json").read_text())
        return {"coeff": coeff, "scaler": scaler, "calib": calib, "meta": meta}
    except Exception as e:
        print(f"  Model load failed: {e}")
        return None


def predict_proba(model: dict, feature_vec: dict) -> float:
    """Logistic regression sigmoid with isotonic calibration."""
    features  = model["meta"]["feature_names"]
    coeff_map = model["coeff"]   # dict: {feature: weight, _intercept: value}
    intercept = coeff_map.get("_intercept", coeff_map.get("intercept", 0.0))
    mean      = model["scaler"]["mean"]
    scale     = model["scaler"]["scale"]

    x = np.array([(feature_vec.get(f, 0.0) - mean[i]) / (scale[i] if scale[i] != 0 else 1.0)
                  for i, f in enumerate(features)])
    coeff = np.array([coeff_map.get(f, 0.0) for f in features])
    logit = float(np.dot(coeff, x)) + intercept
    raw   = 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, logit))))

    # Isotonic calibration: linear interpolation between thresholds
    calib = model["calib"]
    # Support both {bins/calibrated} and {x_thresholds/y_thresholds} formats
    bins = calib.get("bins", calib.get("x_thresholds", []))
    cals = calib.get("calibrated", calib.get("y_thresholds", []))
    if not bins or not cals:
        return raw
    if raw <= bins[0]:
        return cals[0]
    if raw >= bins[-1]:
        return cals[-1]
    for i in range(len(bins) - 1):
        if bins[i] <= raw <= bins[i + 1]:
            t = (raw - bins[i]) / (bins[i + 1] - bins[i])
            return cals[i] + t * (cals[i + 1] - cals[i])
    return raw


# ── Feature builder ────────────────────────────────────────────────────────────

def build_features(game: dict, elo_ratings: dict,
                   h_stats: dict, a_stats: dict) -> dict:
    hid, aid  = game["home_id"], game["away_id"]
    is_neutral = float(game["neutral"])
    is_home    = 1.0 - is_neutral

    rh = elo_ratings.get(game["home_abbr"], elo_ratings.get(hid, INITIAL_ELO))
    ra = elo_ratings.get(game["away_abbr"], elo_ratings.get(aid, INITIAL_ELO))
    elo_diff = rh - ra

    def eff(stats: dict):
        pts_for  = stats.get("avgPointsFor",  75.0)
        pts_ag   = stats.get("avgPointsAgainst", 75.0)
        gp       = max(1, stats.get("gamesPlayed", 1))
        wins     = stats.get("wins", gp / 2)
        oe       = (pts_for  / CBB_TEMPO) * 100
        de       = (pts_ag   / CBB_TEMPO) * 100
        em       = oe - de
        wp       = wins / gp
        return {"oe": oe, "de": de, "em": em, "wp": wp}

    h = eff(h_stats)
    a = eff(a_stats)

    h_pts = h_stats.get("avgPointsFor", 75.0)
    a_pts = a_stats.get("avgPointsFor", 75.0)
    h_opp = h_stats.get("avgPointsAgainst", 75.0)
    a_opp = a_stats.get("avgPointsAgainst", 75.0)

    return {
        "elo_diff":                    elo_diff,
        # Playoff model features
        "win_pct_diff":                h["wp"] - a["wp"],
        "ppg_diff":                    h_pts - a_pts,
        "papg_diff":                   h_opp - a_opp,
        "is_neutral":                  is_neutral,
        # Regular model features
        "adj_em_diff":                 h["em"] - a["em"],
        "adj_oe_diff":                 h["oe"] - a["oe"],
        "adj_de_diff":                 h["de"] - a["de"],
        "adj_tempo_diff":              0.0,
        "pythagorean_diff":            h["wp"] - a["wp"],
        "barthag_diff":                h["wp"] - a["wp"],
        "recruiting_composite_diff":   0.0,
        "returning_minutes_diff":      0.0,
        "transfer_portal_impact_diff": 0.0,
        "experience_diff":             0.0,
        "sos_diff":                    0.0,
        "efg_pct_diff":                0.0,
        "efg_allowed_diff":            0.0,
        "tov_pct_diff":                0.0,
        "tov_forced_diff":             0.0,
        "oreb_pct_diff":               0.0,
        "ft_rate_diff":                0.0,
        "three_pt_pct_diff":           0.0,
        "two_pt_pct_diff":             0.0,
        "block_pct_diff":              0.0,
        "steal_pct_diff":              0.0,
        "team_recent_em_diff":         0.0,
        "rest_days_diff":              0.0,
        "b2b_flag":                    0.0,
        "is_home":                     is_home,
        "is_neutral_site":             is_neutral,
        "home_court_advantage":        is_home,
        "injury_impact_diff":          0.0,
        "vegas_home_prob":             0.0,
        "mc_win_pct":                  0.0,
    }


# ── Printing ───────────────────────────────────────────────────────────────────

def pad(s: str, w: int) -> str:
    return s[:w].ljust(w)


def print_predictions(results: list, date_str: str) -> None:
    width = 95
    print("\n" + "=" * width)
    print(f"  NCAAM ORACLE v4.1  |  {date_str}  |  {len(results)} games")
    print("=" * width)
    header = "  " + pad("MATCHUP", 32) + pad("HOME WIN%", 11) + pad("AWAY WIN%", 11) + pad("ELO DIFF", 10) + "PICK"
    print(header)
    print("-" * width)
    for r in sorted(results, key=lambda x: -max(x["home_prob"], x["away_prob"])):
        matchup    = f"{r['home_abbr']} vs {r['away_abbr']}"
        home_pct   = f"{r['home_prob']*100:.1f}%"
        away_pct   = f"{r['away_prob']*100:.1f}%"
        elo_str    = f"{r['elo_diff']:+.0f}"
        pick       = r["home_abbr"] if r["home_prob"] >= r["away_prob"] else r["away_abbr"]
        star       = " *" if max(r["home_prob"], r["away_prob"]) >= 0.70 else ""
        neutral_tag = " [N]" if r["neutral"] else ""
        print(f"  {pad(matchup + neutral_tag, 32)}{pad(home_pct, 11)}{pad(away_pct, 11)}{pad(elo_str, 10)}{pick}{star}")
    print("-" * width)
    print("* = high confidence (>= 70%)  |  [N] = neutral site\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    args = parser.parse_args()
    date_str = args.date

    print(f"=== NCAAM Oracle v4.1 — Predictions for {date_str} ===\n")

    # Load model
    model = load_model()
    if not model:
        print("ERROR: Could not load model. Run: python python/train_model.py")
        return

    # Build Elo from history
    print("Loading Elo ratings from historical data...")
    elo_ratings = build_elo_from_history()
    print(f"  {len(elo_ratings)} teams have Elo ratings")

    # Fetch today's games; fall back to nearest date with games (±7 days)
    print(f"\nFetching games for {date_str}...")
    games = fetch_games(date_str)

    if not games:
        for offset in list(range(1, 8)) + list(range(-1, -8, -1)):
            d = (datetime.strptime(date_str, "%Y%m%d") + timedelta(days=offset)).strftime("%Y%m%d")
            games = fetch_games(d)
            if games:
                date_str = d
                label = "next" if offset > 0 else "most recent"
                print(f"  No games today — showing {label} games ({d})")
                break

    if not games:
        # Last resort: no-date query returns ESPN's current active game window
        url = f"{ESPN_BASE}/scoreboard?groups=50&limit=200"
        data = fetch_json(url)
        if data:
            games = []
            for event in data.get("events", []):
                comp = (event.get("competitions") or [{}])[0]
                competitors = comp.get("competitors", [])
                home = next((c for c in competitors if c.get("homeAway") == "home"), None)
                away = next((c for c in competitors if c.get("homeAway") == "away"), None)
                if not home or not away:
                    continue
                games.append({
                    "event_id":   event.get("id", ""),
                    "event_name": event.get("name", ""),
                    "status":     event.get("status", {}).get("type", {}).get("name", ""),
                    "date":       event.get("date", "")[:10],
                    "neutral":    int(comp.get("neutralSite", False)),
                    "home_id":    home.get("team", {}).get("id", ""),
                    "home_abbr":  home.get("team", {}).get("abbreviation", "").upper(),
                    "home_name":  home.get("team", {}).get("displayName", ""),
                    "away_id":    away.get("team", {}).get("id", ""),
                    "away_abbr":  away.get("team", {}).get("abbreviation", "").upper(),
                    "away_name":  away.get("team", {}).get("displayName", ""),
                })
            if games:
                date_str = games[0]["date"]
                print(f"  Using ESPN's current window — {date_str}")

    if not games:
        print("No games found in the current window. Season may be over.")
        return

    scheduled = [g for g in games if "SCHEDULED" in g["status"] or "STATUS_SCHEDULED" in g["status"]]
    if not scheduled:
        scheduled = games  # show all if none strictly scheduled (e.g. in-progress or just finished)

    print(f"  Found {len(scheduled)} game(s)\n")

    # Predict each game
    results = []
    for game in scheduled:
        h_stats = fetch_team_stats(game["home_id"])
        a_stats = fetch_team_stats(game["away_id"])
        time.sleep(0.1)

        fv       = build_features(game, elo_ratings, h_stats, a_stats)
        home_p   = predict_proba(model, fv)
        away_p   = 1.0 - home_p

        results.append({
            "home_abbr": game["home_abbr"],
            "away_abbr": game["away_abbr"],
            "home_name": game["home_name"],
            "away_name": game["away_name"],
            "home_prob": home_p,
            "away_prob": away_p,
            "elo_diff":  fv["elo_diff"],
            "neutral":   game["neutral"],
            "status":    game["status"],
        })

    print_predictions(results, date_str)


if __name__ == "__main__":
    main()
