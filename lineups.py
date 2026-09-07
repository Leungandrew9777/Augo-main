"""
lineups.py — injury / suspension availability features (implementation method).

Why
---
Missing key players are the biggest real-world signal the current pipeline
lacks: ELO is only updated from full-time results, so a team missing its top
scorer keeps its ELO until the next matchday. Availability data cannot be
trained into the model (no historical injury dataset is freely available), so
the method below applies a *post-hoc* adjustment instead.

Method
------
1. SOURCE — API-Football v3 (https://www.api-football.com), free tier:
     GET /teams?league=39&season={year}        -> Premier League team ids
     GET /injuries?team={id}&season={year}     -> injuries + suspensions
   (Football-Data.co.uk and Understat do not publish availability data.)
   Requires ``APIFOOTBALL_KEY`` in .env; the fetch is skipped without it.

2. IMPACT — each unavailable player maps to an impact score by position
   (goal-adjacent positions weigh more):
       GK=2.0  DF=1.5  MF=1.0  FW=2.0

3. ELO PENALTY — per team: ``min(impact_sum, IMPACT_CAP) * ELO_PER_IMPACT``
   (defaults: cap 6.0, 15 ELO per impact unit => up to ~90 ELO, typical
   key-player absence ≈ 15-45 ELO).

4. ADJUSTMENT — before running the model, subtract the penalty from each
   side's ELO and recompute ``elo_diff``. The stacked model is feature-based,
   so shifting ``elo_diff`` and re-predicting gives calibrated probs for the
   weakened side with no retraining.

5. AUDIT — ``disp_availability`` columns are added to the cache so you can see
   what was applied.

Enable with::

    AUGO_ENABLE_LINEUPS=1
    APIFOOTBALL_KEY=<your_key>

in .env. Without the flag, a key, or a successful fetch the pipeline runs
unchanged.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

LEAGUE_ID = 39            # API-Football: Premier League
API_BASE = "https://v3.football.api-sports.io"

POSITION_IMPACT: dict[str, float] = {"GK": 2.0, "DF": 1.5, "MF": 1.0, "FW": 2.0}
IMPACT_CAP = 6.0
ELO_PER_IMPACT = 15.0

_HEADERS = {"x-apisports-key": "", "User-Agent": "augo/1.0"}


def _enabled() -> bool:
    return os.getenv("AUGO_ENABLE_LINEUPS", "").strip() == "1"


def _session():
    import requests

    key = os.getenv("APIFOOTBALL_KEY", "").strip()
    if not key or key == "your_key_here":
        return None
    s = requests.Session()
    s.headers.update({**_HEADERS, "x-apisports-key": key})
    return s


def _short_name(api_name: str) -> str:
    from team_aliases import fixture_lookup_key
    return fixture_lookup_key(str(api_name))


def fetch_team_ids(session, season_year: int) -> dict[str, int]:
    """{short_team_name: api_football_team_id} for the current season."""
    import requests

    r = session.get(f"{API_BASE}/teams", params={"league": LEAGUE_ID, "season": season_year}, timeout=20)
    r.raise_for_status()
    out: dict[str, int] = {}
    for item in r.json().get("response", []):
        team = item.get("team", {})
        name = _short_name(team.get("name", ""))
        tid = team.get("id")
        if name and tid is not None:
            out[name] = int(tid)
    return out


def fetch_injuries(session, season_year: int) -> dict[str, list[dict[str, Any]]]:
    """{short_team_name: [{player, position, type}]} — best-effort fetch."""
    team_ids = fetch_team_ids(session, season_year)
    out: dict[str, list[dict[str, Any]]] = {}
    for team, tid in team_ids.items():
        try:
            r = session.get(f"{API_BASE}/injuries", params={"team": tid, "season": season_year}, timeout=20)
            r.raise_for_status()
        except Exception:
            continue
        for item in r.json().get("response", []):
            player = item.get("player", {}) or {}
            info = item.get("player_information", {}) or {}
            ptype = str(info.get("type", "")).lower()
            if "injur" not in ptype and "suspen" not in ptype:
                continue
            out.setdefault(team, []).append({
                "player": player.get("name", "unknown"),
                "position": str(info.get("position", "MF")),
                "type": ptype,
            })
    return out


def team_penalty(injuries: list[dict[str, Any]]) -> float:
    """ELO-equivalent penalty for a list of unavailable players."""
    impact = 0.0
    for inj in injuries:
        pos = str(inj.get("position", "MF")).upper()[:2]
        impact += POSITION_IMPACT.get(pos, 1.0)
    return float(min(impact, IMPACT_CAP) * ELO_PER_IMPACT)


def apply_availability(
    upcoming: pd.DataFrame,
    injuries: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    """Subtract each side's availability penalty from its ELO, recompute diff.

    Adds ``disp_availability_*`` audit columns. Never errors on bad data —
    returns the frame unchanged if nothing applies.
    """
    out = upcoming.copy()
    if not injuries:
        return out

    home_penalty = out["home_team"].map(lambda t: team_penalty(injuries.get(str(t), [])))
    away_penalty = out["away_team"].map(lambda t: team_penalty(injuries.get(str(t), [])))
    if home_penalty.fillna(0.0).sum() == 0.0 and away_penalty.fillna(0.0).sum() == 0.0:
        return out

    out["elo_home"] = pd.to_numeric(out["elo_home"], errors="coerce").fillna(1500.0) - home_penalty
    out["elo_away"] = pd.to_numeric(out["elo_away"], errors="coerce").fillna(1500.0) - away_penalty
    out["elo_diff"] = out["elo_home"] - out["elo_away"]
    out["disp_availability_home"] = home_penalty.map(lambda v: f"-{v:.0f} ELO" if v else "")
    out["disp_availability_away"] = away_penalty.map(lambda v: f"-{v:.0f} ELO" if v else "")
    return out


def apply_availability_if_enabled(upcoming: pd.DataFrame, season_year: int | None = None) -> pd.DataFrame:
    """Pipeline hook: fetch + apply injuries only when enabled and keyed."""
    if not _enabled():
        return upcoming
    session = _session()
    if session is None:
        print("   ℹ️  APIFOOTBALL_KEY not set — skipping lineup adjustment.")
        return upcoming
    try:
        if season_year is None:
            season_year = int(pd.Timestamp.now().year)
        injuries = fetch_injuries(session, season_year)
        if not injuries:
            print("   ℹ️  No injury data returned — skipping lineup adjustment.")
            return upcoming
        print(f"   ℹ️  Applied lineup adjustment for {len(injuries)} teams.")
        return apply_availability(upcoming, injuries)
    except Exception as exc:
        print(f"   ⚠️  Lineup adjustment skipped: {exc}")
        return upcoming
