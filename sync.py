#!/usr/bin/env python3
"""
sync.py — keep fixtures.csv and results.csv in sync with live data
(The Odds API, same key as run_pipeline.py).

Usage:
    python sync.py fixtures [--gw N] [--days 10]   # refresh/add upcoming fixtures
    python sync.py results  [--days 3]             # append finished match scores
    python sync.py all

Notes
-----
* Fixtures: existing (date, home, away) rows are kept; matching team pairings
  get their date refreshed (postponements); brand-new events are grouped into
  date clusters and assigned the next GW number(s) after the current max
  (use ``--gw N`` to start a new season at GW N).
* Results: only matches whose (date/team) pairing is found in fixtures.csv are
  written, so gameweeks are always correct; rows already present are updated
  in place (e.g. goal corrections).
* Override output paths with AUGO_FIXTURES_FILE / AUGO_RESULTS_FILE.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

from team_aliases import badge_lookup_key, fixture_lookup_key

load_dotenv()

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_FILE = os.getenv("AUGO_FIXTURES_FILE") or os.path.join(APP_DIR, "fixtures.csv")
RESULTS_FILE = os.getenv("AUGO_RESULTS_FILE") or os.path.join(APP_DIR, "results.csv")

SPORT = "soccer_epl"
BASE = "https://api.the-odds-api.com/v4/sports"

FIXTURE_COLS = ["gameweek", "date", "home_team", "away_team"]
RESULT_COLS = ["date", "gameweek", "home_team", "away_team", "home_goals", "away_goals"]


def _api_key() -> str | None:
    key = os.getenv("ODDS_API_KEY", "").strip()
    if not key or key == "your_key_here":
        return None
    return key


def _iso(dt_str: str) -> str:
    return str(dt_str)[:10]


def _write_fmt(dt_str: str) -> str:
    """ISO YYYY-MM-DD -> DD/MM/YYYY (results.csv / fixtures.csv style)."""
    ts = pd.to_datetime(dt_str, errors="coerce")
    if pd.isna(ts):
        return dt_str
    return f"{ts.day}/{ts.month}/{ts.year}"


# ── API ──────────────────────────────────────────────────────────────────────

def fetch_upcoming_events(days: int = 10) -> pd.DataFrame:
    key = _api_key()
    if key is None:
        print("⚠️  ODDS_API_KEY not set — cannot fetch fixtures.")
        return pd.DataFrame()
    now = datetime.now(timezone.utc)
    params = {
        "apiKey": key,
        "regions": "uk",
        "markets": "h2h",
        "commenceTimeFrom": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commenceTimeTo": (now + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        resp = requests.get(f"{BASE}/{SPORT}/odds", params=params, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        print(f"⚠️  Odds API fixtures request failed: {exc}")
        return pd.DataFrame()
    rows = [
        {
            "date": _iso(ev.get("commence_time", "")),
            "home_team": fixture_lookup_key(ev.get("home_team", "")),
            "away_team": fixture_lookup_key(ev.get("away_team", "")),
        }
        for ev in resp.json()
    ]
    out = pd.DataFrame(rows, columns=["date", "home_team", "away_team"])
    return out.drop_duplicates().dropna(subset=["date", "home_team", "away_team"])


def fetch_finished_scores(days: int = 3) -> pd.DataFrame:
    key = _api_key()
    if key is None:
        print("⚠️  ODDS_API_KEY not set — cannot fetch results.")
        return pd.DataFrame()
    try:
        resp = requests.get(f"{BASE}/{SPORT}/scores", params={"apiKey": key, "daysFrom": days}, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        print(f"⚠️  Odds API scores request failed: {exc}")
        return pd.DataFrame()
    rows = []
    for ev in resp.json():
        if not ev.get("completed"):
            continue
        score_map = {s.get("name"): s.get("score") for s in ev.get("scores", [])}
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        hg, ag = score_map.get(home), score_map.get(away)
        if hg is None or ag is None:
            continue
        rows.append({
            "date": _iso(ev.get("commence_time", "")),
            "home_team": fixture_lookup_key(home),
            "away_team": fixture_lookup_key(away),
            "home_goals": int(hg),
            "away_goals": int(ag),
        })
    out = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_goals", "away_goals"])
    return out.drop_duplicates(subset=["date", "home_team", "away_team"])


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _load_fixtures() -> pd.DataFrame:
    if not os.path.exists(FIXTURES_FILE):
        return pd.DataFrame(columns=FIXTURE_COLS)
    df = pd.read_csv(FIXTURES_FILE)
    for col in FIXTURE_COLS:
        if col not in df.columns:
            df[col] = ""
    df = df[FIXTURE_COLS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True).dt.strftime("%Y-%m-%d")
    df["gameweek"] = pd.to_numeric(df["gameweek"], errors="coerce")
    return df.dropna(subset=["date", "home_team", "away_team"]).reset_index(drop=True)


def sync_fixtures(days: int = 10, gw_override: int | None = None) -> None:
    events = fetch_upcoming_events(days)
    if events.empty:
        print("   No upcoming events returned — fixtures.csv unchanged.")
        return

    existing = _load_fixtures()
    rows = existing.to_dict("records")
    max_gw = int(existing["gameweek"].max()) if len(existing) and existing["gameweek"].notna().any() else 0

    added = 0
    updated = 0
    new_events: list[dict] = []
    for ev in events.to_dict("records"):
        h, a = str(ev["home_team"]).strip(), str(ev["away_team"]).strip()
        exact = [r for r in rows if str(r["home_team"]).strip() == h and str(r["away_team"]).strip() == a and str(r["date"]) == str(ev["date"])]
        if exact:
            continue
        # Single *future* pairing -> refresh its date (postponement/rearrange).
        # Past rows are never touched, so historical gameweeks stay intact.
        today = _iso(datetime.now().isoformat())
        pair = [r for r in rows if str(r["home_team"]).strip() == h and str(r["away_team"]).strip() == a]
        if len(pair) == 1 and pd.notna(pair[0].get("gameweek")) and str(pair[0]["date"]) >= today:
            pair[0]["date"] = str(ev["date"])
            updated += 1
        else:
            new_events.append(ev)

    if new_events:
        base = int(gw_override) if gw_override is not None else max_gw + 1
        clusters: list[list[dict]] = []
        cur: list[dict] = []
        prev: datetime | None = None
        for ev in sorted(new_events, key=lambda x: x["date"]):
            d = datetime.strptime(str(ev["date"]), "%Y-%m-%d")
            if prev is not None and (d - prev).days > 1:
                clusters.append(cur)
                cur = []
            cur.append(ev)
            prev = d
        if cur:
            clusters.append(cur)
        for i, cluster in enumerate(clusters):
            gw = base + i
            for ev in cluster:
                rows.append({"gameweek": gw, "date": str(ev["date"]),
                             "home_team": str(ev["home_team"]).strip(), "away_team": str(ev["away_team"]).strip()})
                added += 1
        print(f"   Assigned GW {base}..{base + len(clusters) - 1} to {added} new fixture(s)"
              f" (override with --gw N for a new season).")

    out = pd.DataFrame(rows, columns=FIXTURE_COLS)
    out = out.drop_duplicates(subset=["home_team", "away_team", "date"])
    out["date"] = out["date"].map(_write_fmt)
    out = out.sort_values(["date", "gameweek"])
    out.to_csv(FIXTURES_FILE, index=False)
    print(f"fixtures: {added} added, {updated} date-updated -> {len(out)} total ({FIXTURES_FILE})")


# ── Results ───────────────────────────────────────────────────────────────────

def _load_results() -> pd.DataFrame:
    if not os.path.exists(RESULTS_FILE):
        return pd.DataFrame(columns=RESULT_COLS)
    df = pd.read_csv(RESULTS_FILE)
    for col in RESULT_COLS:
        if col not in df.columns:
            df[col] = ""
    return df[RESULT_COLS].copy()


def sync_results(days: int = 3) -> None:
    scores = fetch_finished_scores(days)
    if scores.empty:
        print("   No completed matches returned — results.csv unchanged.")
        return

    fixtures = _load_fixtures()
    if fixtures.empty:
        print("   fixtures.csv is empty — cannot map results to gameweeks.")
        return
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce", dayfirst=True).dt.strftime("%Y-%m-%d")
    date_gw = {
        (str(r["date"]), str(r["home_team"]).strip(), str(r["away_team"]).strip()): int(r["gameweek"])
        for _, r in fixtures.iterrows() if pd.notna(r["gameweek"])
    }
    pair_gw: dict[tuple[str, str], list[int]] = {}
    for _, r in fixtures.iterrows():
        if pd.notna(r["gameweek"]):
            pair_gw.setdefault((str(r["home_team"]).strip(), str(r["away_team"]).strip()), []).append(int(r["gameweek"]))

    results = _load_results()
    existing: dict[tuple[str, str], int] = {}
    for i, r in results.iterrows():
        key = (fixture_lookup_key(str(r["home_team"])), fixture_lookup_key(str(r["away_team"])))
        existing.setdefault(key, i)

    added = 0
    updated = 0
    skipped: list[str] = []
    for s in scores.to_dict("records"):
        h, a = str(s["home_team"]).strip(), str(s["away_team"]).strip()
        gw = date_gw.get((str(s["date"]), h, a))
        if gw is None:
            gws = pair_gw.get((h, a), [])
            gw = gws[0] if len(gws) == 1 else None
        if gw is None:
            skipped.append(f"{h} v {a} ({s['date']})")
            continue
        row = {
            "date": _write_fmt(str(s["date"])),
            "gameweek": gw,
            "home_team": badge_lookup_key(h),
            "away_team": badge_lookup_key(a),
            "home_goals": int(s["home_goals"]),
            "away_goals": int(s["away_goals"]),
        }
        key = (h, a)
        if key in existing:
            idx = existing[key]
            old = results.iloc[idx]
            if int(old["home_goals"]) != row["home_goals"] or int(old["away_goals"]) != row["away_goals"]:
                results.iloc[idx] = pd.Series(row)
                updated += 1
        else:
            results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
            existing[key] = len(results) - 1
            added += 1

    if skipped:
        print(f"   ⚠️  Skipped {len(skipped)} match(es) not in fixtures.csv: {', '.join(skipped[:5])}"
              + (" …" if len(skipped) > 5 else ""))
    out = results.drop_duplicates(subset=["home_team", "away_team"])
    out = out.sort_values(["date", "gameweek"])
    out.to_csv(RESULTS_FILE, index=False)
    print(f"results: {added} added, {updated} updated -> {len(out)} total ({RESULTS_FILE})")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fix = sub.add_parser("fixtures")
    p_fix.add_argument("--gw", type=int, default=None, help="force GW for new fixtures (new season)")
    p_fix.add_argument("--days", type=int, default=10)

    p_res = sub.add_parser("results")
    p_res.add_argument("--days", type=int, default=3)

    sub.add_parser("all")
    args = parser.parse_args()

    if args.cmd in ("fixtures", "all"):
        sync_fixtures(days=getattr(args, "days", 10), gw_override=getattr(args, "gw", None))
    if args.cmd in ("results", "all"):
        sync_results(days=getattr(args, "days", 3))


if __name__ == "__main__":
    main()
