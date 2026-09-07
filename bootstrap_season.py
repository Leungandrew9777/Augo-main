#!/usr/bin/env python3
"""
bootstrap_season.py — bootstrap the new Premier League season into the training
data when football-data.co.uk is not yet available for it.

Situation (2026/27 after GW3): football-data.co.uk 2627/E0.csv is 503, but
Understat publishes the new season (matches + real xG) and The Odds API has the
results. This script appends the finished new-season matches to
``premier_league_historical_clean.csv`` so feature_engineering.py + train_ensemble.py
can rebuild ELO/form features that already include the new season.

What Understat lacks (shots, corners, cards, bookmaker odds) is filled with
per-column league medians from the existing data, so the shot/corner features
are neutral (real xG still overrides the proxy). Idempotent: skips if the
season is already present.

Usage:
    python bootstrap_season.py [--season 2627] [--understat-year 2026]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from data_pipeline import UNDERSTAT_TO_SHORT

APP_DIR = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
CLEAN_FILE = f"{APP_DIR}/premier_league_historical_clean.csv"

# Columns football-data provides but Understat does not: filled with medians.
STAT_COLS = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
ODDS_COLS = ["B365H", "B365D", "B365A"]

RESULT_MAP = {"H": 2, "D": 1, "A": 0}


def _short(title: str) -> str:
    return UNDERSTAT_TO_SHORT.get(str(title).strip(), str(title).strip())


def fetch_matches_with_goals(year: int) -> pd.DataFrame:
    """Understat dates payload -> date, home_short, away_short, FTHG, FTAG, xG."""
    import json

    import requests

    sess = requests.Session()
    sess.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json, text/javascript, */*; q=0.01",
    })
    sess.get(f"https://understat.com/league/EPL/{year}", timeout=25)
    resp = sess.get(f"https://understat.com/main/getLeagueData/EPL/{year}", timeout=25)
    resp.raise_for_status()
    data = json.loads(resp.text)

    rows = []
    for m in data.get("dates", []):
        if not m.get("isResult"):
            continue
        try:
            hg, ag = int(m["goals"]["h"]), int(m["goals"]["a"])
            hxg, axg = float(m["xG"]["h"]), float(m["xG"]["a"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({
            "date": str(m.get("datetime", ""))[:10],
            "home_team": _short(m["h"]["title"]),
            "away_team": _short(m["a"]["title"]),
            "home_goals": hg,
            "away_goals": ag,
            "home_xg": hxg,
            "away_xg": axg,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2627, help="football-data season code")
    parser.add_argument("--understat-year", type=int, default=2026, help="understat season start year")
    args = parser.parse_args()

    df = pd.read_csv(CLEAN_FILE)
    if args.season in set(df["Season"].astype(int).unique()):
        print(f"Season {args.season} already present — nothing to do.")
        return

    matches = fetch_matches_with_goals(args.understat_year)
    if matches.empty:
        print("⚠️  No finished new-season matches returned by Understat.")
        return

    # League medians for the stats Understat does not provide (neutral fill).
    medians: dict[str, float] = {}
    for col in STAT_COLS:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if not s.empty:
                medians[col] = float(s.median())

    rows = []
    for _, m in matches.iterrows():
        ftr = "H" if m["home_goals"] > m["away_goals"] else ("A" if m["home_goals"] < m["away_goals"] else "D")
        row = {
            "Date": m["date"],
            "HomeTeam": m["home_team"],
            "AwayTeam": m["away_team"],
            "FTHG": m["home_goals"],
            "FTAG": m["away_goals"],
            "FTR": ftr,
            "HTHG": np.nan, "HTAG": np.nan, "HTR": np.nan,
        }
        for col in STAT_COLS:
            row[col] = medians.get(col, np.nan)
        for col in ODDS_COLS:
            row[col] = np.nan
        row.update({
            "League": "Premier League",
            "Season": args.season,
            "Result": RESULT_MAP[ftr],
            "home_xg": m["home_xg"],
            "away_xg": m["away_xg"],
        })
        rows.append(row)

    new = pd.DataFrame(rows)
    missing = [c for c in df.columns if c not in new.columns]
    for c in missing:
        new[c] = np.nan
    new = new[df.columns.tolist()]

    out = pd.concat([df, new], ignore_index=True)
    out.to_csv(CLEAN_FILE, index=False)
    print(f"✓ Appended {len(new)} new-season matches (Season {args.season}) "
          f"-> {len(out)} total ({CLEAN_FILE})")
    print(f"  Teams: {sorted(set(new['HomeTeam']).union(set(new['AwayTeam'])))}")


if __name__ == "__main__":
    main()
