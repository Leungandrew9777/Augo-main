"""
persistence.py — disk-backed history layer for the Augo app.

Three pieces:
  - load_archived_predictions(): every predictions_history/GW{N}.json
  - load_results(): {(gw, home, away): "H"/"D"/"A"} from results.csv
  - load_user_picks() / save_user_picks(): {gw: {match_idx: "H"/"D"/"A"}} on disk

Team names from results.csv are normalized via team_aliases.fixture_lookup_key
so they line up with predictions_cache.json fixture names.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import pandas as pd

from team_aliases import fixture_lookup_key

from data_urls import results_csv_url

APP_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(APP_DIR, "predictions_history")
RESULTS_FILE = os.path.join(APP_DIR, "results.csv")
USER_PICKS_FILE = os.path.join(APP_DIR, "user_picks.json")
BANKROLL_FILE = os.path.join(APP_DIR, "bankroll.json")


def _norm(name: Any) -> str:
    return fixture_lookup_key(str(name).strip())


def _gw_from_filename(name: str) -> int | None:
    m = re.search(r"GW(\d+)", name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def load_archived_predictions() -> dict[int, dict[str, Any]]:
    """Return {gw_number: cache_dict} for every predictions_history/GW*.json."""
    out: dict[int, dict[str, Any]] = {}
    if not os.path.isdir(HISTORY_DIR):
        return out
    for fname in os.listdir(HISTORY_DIR):
        if not fname.lower().endswith(".json"):
            continue
        gw = _gw_from_filename(fname)
        if gw is None:
            continue
        path = os.path.join(HISTORY_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                out[gw] = json.load(f)
        except Exception:
            continue
    return out


def _result_letter(home_goals: Any, away_goals: Any) -> str | None:
    try:
        h = int(home_goals)
        a = int(away_goals)
    except (TypeError, ValueError):
        return None
    if h > a:
        return "H"
    if h < a:
        return "A"
    return "D"


def _results_dataframe() -> pd.DataFrame | None:
    """Load results from env, remote_data_urls.json, or results.csv."""
    url = results_csv_url()
    if url:
        try:
            return pd.read_csv(url)
        except Exception:
            pass
    if not os.path.exists(RESULTS_FILE):
        return None
    try:
        return pd.read_csv(RESULTS_FILE)
    except Exception:
        return None


def load_results() -> dict[tuple[int, str, str], dict[str, Any]]:
    """Return {(gw, home_short, away_short): {actual, home_goals, away_goals}}.

    Reads results.csv, or a public CSV URL from env / remote_data_urls.json
    when set (for deployed apps). Missing / invalid -> empty dict.
    """
    out: dict[tuple[int, str, str], dict[str, Any]] = {}
    df = _results_dataframe()
    if df is None:
        return out
    needed = {"gameweek", "home_team", "away_team", "home_goals", "away_goals"}
    if not needed.issubset(df.columns):
        return out
    for _, row in df.iterrows():
        try:
            gw = int(row["gameweek"])
        except (TypeError, ValueError):
            continue
        home = _norm(row["home_team"])
        away = _norm(row["away_team"])
        actual = _result_letter(row.get("home_goals"), row.get("away_goals"))
        if actual is None:
            continue
        out[(gw, home, away)] = {
            "actual": actual,
            "home_goals": int(row["home_goals"]),
            "away_goals": int(row["away_goals"]),
        }
    return out


def load_user_picks() -> dict[int, dict[int, str]]:
    """Return {gw: {match_idx: 'H'/'D'/'A'}} from user_picks.json (or {})."""
    if not os.path.exists(USER_PICKS_FILE):
        return {}
    try:
        with open(USER_PICKS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    out: dict[int, dict[int, str]] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        try:
            gw = int(k)
        except (TypeError, ValueError):
            continue
        if not isinstance(v, dict):
            continue
        bucket: dict[int, str] = {}
        for mk, mv in v.items():
            try:
                idx = int(mk)
            except (TypeError, ValueError):
                continue
            if isinstance(mv, str) and mv in ("H", "D", "A"):
                bucket[idx] = mv
        out[gw] = bucket
    return out


def save_user_picks(picks_by_gw: dict[int, dict[int, str]]) -> None:
    """Persist the picks map to disk (atomic write)."""
    serializable: dict[str, dict[str, str]] = {
        str(int(gw)): {str(int(idx)): pick for idx, pick in picks.items() if pick}
        for gw, picks in picks_by_gw.items()
    }
    tmp = USER_PICKS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    os.replace(tmp, USER_PICKS_FILE)


def upsert_user_picks_for_gw(gw: int, picks_by_match_idx: dict[int, str]) -> None:
    """Convenience: load -> set one GW -> save."""
    all_picks = load_user_picks()
    cleaned = {idx: pick for idx, pick in picks_by_match_idx.items() if pick in ("H", "D", "A")}
    all_picks[int(gw)] = cleaned
    save_user_picks(all_picks)


def default_bankroll() -> dict[str, Any]:
    return {
        "starting_bankroll": 1000.0,
        "current_bankroll": 1000.0,
        "risk_cap": 0.05,
        "ledger": [],
    }


def load_bankroll() -> dict[str, Any]:
    if not os.path.exists(BANKROLL_FILE):
        return default_bankroll()
    try:
        with open(BANKROLL_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return default_bankroll()
    base = default_bankroll()
    if isinstance(raw, dict):
        base.update(raw)
    try:
        base["starting_bankroll"] = float(base.get("starting_bankroll", 1000.0))
        base["current_bankroll"] = float(base.get("current_bankroll", base["starting_bankroll"]))
        base["risk_cap"] = float(base.get("risk_cap", 0.05))
    except (TypeError, ValueError):
        base = default_bankroll()
    if not isinstance(base.get("ledger"), list):
        base["ledger"] = []
    return base


def save_bankroll(bankroll: dict[str, Any]) -> None:
    tmp = BANKROLL_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(bankroll, f, indent=2)
    os.replace(tmp, BANKROLL_FILE)
