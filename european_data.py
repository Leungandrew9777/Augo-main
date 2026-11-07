#!/usr/bin/env python3
"""european_data.py — top-5 European results + cross-league Elo + external Elo.

Data sources (free):
  * top-5 domestic leagues: openfootball football.json
  * UCL / UEL / UECL: openfootball champions-league (cl.txt / el.txt / conf.txt)

Outputs (new files only; existing fixtures/results/history are never touched):
  * european_results.csv       all matches, canonical team keys
  * european_elo_history.csv   per-team rating after each match (as-of queries)
  * european_elo_current.csv   latest rating per team
  * european_features.csv      cross-league Elo + UEFA rest features for PL matches
  * external_elo.csv           canonicalised ratings imported from your ClubElo file

The cross-league Elo is our own sequential rating over the combined dataset
(top-5 domestic + UEFA cups), so it is fully historical and backtestable.
ClubElo ratings can be imported periodically (the CSV API is currently down):

    # file columns: team,elo  (optionally date=YYYY-MM-DD for history)
    python european_data.py --import-elo path/to/clubelo.csv

They are used only as a fallback for teams with no history in our dataset and
as an optional feature candidate; the model only keeps them if a rolling-origin
backtest shows an improvement.

Usage:
    python european_data.py            # rebuild results + Elo + PL features
    python european_data.py --report   # print coverage / validation report only
    python european_data.py --import-elo data/external/clubelo_ratings.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import unicodedata
from datetime import date

import pandas as pd
import requests

import dates
from data_quality import HISTORY_SEASONS

APP_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(APP_DIR, "data", "external", "raw")
RESULTS_FILE = os.path.join(APP_DIR, "european_results.csv")
ELO_HISTORY_FILE = os.path.join(APP_DIR, "european_elo_history.csv")
ELO_CURRENT_FILE = os.path.join(APP_DIR, "european_elo_current.csv")
FEATURES_FILE = os.path.join(APP_DIR, "european_features.csv")
EXTERNAL_ELO_FILE = os.path.join(APP_DIR, "external_elo.csv")
CLEAN_FILE = os.path.join(APP_DIR, "premier_league_historical_clean.csv")

OPENFOOTBALL_JSON = "https://raw.githubusercontent.com/openfootball/football.json/master"
OPENFOOTBALL_UEFA = "https://raw.githubusercontent.com/openfootball/champions-league/master"

DOMESTIC_COMPS = {
    "en.1": "EPL",
    "es.1": "LaLiga",
    "de.1": "Bundesliga",
    "it.1": "SerieA",
    "fr.1": "Ligue1",
}
UEFA_COMPS = {"cl": "UCL", "el": "UEL", "conf": "UECL"}
EXPECTED_DOMESTIC_MIN = 250  # Bundesliga/Ligue1 have 306; 20-team leagues 380

# openfootball club name (either season variant) -> football-data clean name.
EURO_TO_CLEAN: dict[str, str] = {
    "AFC Bournemouth": "Bournemouth", "Bournemouth": "Bournemouth",
    "Arsenal": "Arsenal", "Arsenal FC": "Arsenal",
    "Aston Villa": "Aston Villa", "Aston Villa FC": "Aston Villa",
    "Brentford": "Brentford", "Brentford FC": "Brentford",
    "Brighton & Hove Albion": "Brighton", "Brighton & Hove Albion FC": "Brighton",
    "Burnley": "Burnley", "Burnley FC": "Burnley",
    "Cardiff City": "Cardiff", "Cardiff City FC": "Cardiff",
    "Chelsea": "Chelsea", "Chelsea FC": "Chelsea",
    "Coventry City": "Coventry", "Coventry City FC": "Coventry",
    "Crystal Palace": "Crystal Palace", "Crystal Palace FC": "Crystal Palace",
    "Everton": "Everton", "Everton FC": "Everton",
    "Fulham": "Fulham", "Fulham FC": "Fulham",
    "Huddersfield Town": "Huddersfield", "Huddersfield Town AFC": "Huddersfield",
    "Hull City": "Hull", "Hull City AFC": "Hull", "Hull City FC": "Hull",
    "Ipswich Town": "Ipswich", "Ipswich Town FC": "Ipswich",
    "Leeds United": "Leeds", "Leeds United FC": "Leeds",
    "Leicester City": "Leicester", "Leicester City FC": "Leicester",
    "Liverpool": "Liverpool", "Liverpool FC": "Liverpool",
    "Luton Town": "Luton", "Luton Town FC": "Luton",
    "Manchester City": "Man City", "Manchester City FC": "Man City",
    "Manchester United": "Man United", "Manchester United FC": "Man United",
    "Middlesbrough": "Middlesbrough", "Middlesbrough FC": "Middlesbrough",
    "Newcastle United": "Newcastle", "Newcastle United FC": "Newcastle",
    "Norwich City": "Norwich", "Norwich City FC": "Norwich",
    "Nottingham Forest": "Nott'm Forest", "Nottingham Forest FC": "Nott'm Forest",
    "Queens Park Rangers": "QPR", "Queens Park Rangers FC": "QPR", "QPR": "QPR",
    "Sheffield United": "Sheffield United", "Sheffield United FC": "Sheffield United",
    "Southampton": "Southampton", "Southampton FC": "Southampton",
    "Stoke City": "Stoke", "Stoke City FC": "Stoke",
    "Sunderland": "Sunderland", "Sunderland AFC": "Sunderland",
    "Swansea City": "Swansea", "Swansea City AFC": "Swansea",
    "Tottenham Hotspur": "Tottenham", "Tottenham Hotspur FC": "Tottenham",
    "Watford": "Watford", "Watford FC": "Watford",
    "West Bromwich Albion": "West Brom", "West Bromwich Albion FC": "West Brom",
    "West Ham United": "West Ham", "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers": "Wolves", "Wolverhampton Wanderers FC": "Wolves",
}

# ClubElo / common spellings -> our clean names (PL only; others stay as-is).
CLUBELO_TO_CLEAN: dict[str, str] = {
    "Man City": "Man City", "Man United": "Man United", "Newcastle": "Newcastle",
    "Nott'm Forest": "Nott'm Forest", "Nottingham Forest": "Nott'm Forest",
    "Tottenham": "Tottenham", "West Ham": "West Ham", "Wolves": "Wolves",
    "Brighton": "Brighton", "Bournemouth": "Bournemouth", "Leeds": "Leeds",
    "Coventry": "Coventry", "Hull": "Hull", "Ipswich": "Ipswich",
}

MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1
)}
DATE_HEADER_RE = re.compile(
    r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$"
)
MATCH_LINE_RE = re.compile(
    r"^\s*(?:\d{1,2}:\d{2}\s+)?(.+?)\s+\(([A-Z]{2,4})\)\s+v\s+(.+?)\s+\(([A-Z]{2,4})\)\s+(.+)$"
)
SCORE_RE = re.compile(r"(\d+)\s*-\s*(\d+)")
AET_RE = re.compile(r"(\d+)\s*-\s*(\d+)\s+a\.e\.t\.")
HEADER_MATCHES_RE = re.compile(r"#\s*Matches\s+(\d+)")

RETRIES = 3
TIMEOUT = 30


def season_folder(season: str) -> str:
    text = str(season)
    return f"20{text[:2]}-{text[2:]}"


def _cache_path(*parts: str) -> str:
    path = os.path.join(RAW_DIR, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _fetch(url: str, cache: str, *, refresh: bool = False) -> str:
    if os.path.exists(cache) and not refresh:
        with open(cache, "r", encoding="utf-8") as f:
            return f.read()
    last_error: Exception | None = None
    for attempt in range(RETRIES):
        try:
            resp = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Augo/1.0"})
            resp.raise_for_status()
            text = resp.text
            with open(cache, "w", encoding="utf-8") as f:
                f.write(text)
            return text
        except Exception as exc:  # noqa: BLE001 - retry any transport error
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}: {last_error}")


def canonical_team(name: str) -> str:
    """Normalise an openfootball club name to a stable cross-league key."""
    raw = str(name).strip()
    if not raw:
        return raw
    if raw in EURO_TO_CLEAN:
        return EURO_TO_CLEAN[raw]
    base = re.sub(r"\s+(FC|AFC|CF|AC|AS|SS|SSC|SC|SV|VfB|VfL|TSG|FSV|BSC)$", "", raw, flags=re.I).strip()
    if base in EURO_TO_CLEAN:
        return EURO_TO_CLEAN[base]
    # Strip a leading club-type token and collapse whitespace for non-PL clubs.
    key = re.sub(r"^(?:\d+\.\s*)?(FC|AFC|CF|AC|AS|SS|SSC|SC|SV|VfB|VfL|TSG|FSV|BSC)\s+", "", base, flags=re.I)
    key = unicodedata.normalize("NFC", re.sub(r"\s+", " ", key).strip())
    return key


def clubelo_key(name: str) -> str:
    raw = str(name).strip()
    return CLUBELO_TO_CLEAN.get(raw, raw)


def _strip_country(name: str) -> str:
    return re.sub(r"\s*\([A-Z]{2,4}\)\s*$", "", str(name)).strip()


# ── parsers ──────────────────────────────────────────────────────────────────

def parse_domestic_json(text: str, season: str, competition: str) -> list[dict]:
    payload = json.loads(text)
    rows: list[dict] = []
    for match in payload.get("matches", []):
        score = match.get("score")
        if isinstance(score, dict):
            ft = score.get("ft")
        elif isinstance(score, (list, tuple)) and len(score) == 2:
            ft = score
        else:
            ft = None
        if not ft or len(ft) != 2:
            continue
        day = dates.iso_date(match.get("date"))
        if not day:
            continue
        rows.append({
            "date": day,
            "competition": competition,
            "season": season,
            "home_team": canonical_team(match["team1"]),
            "away_team": canonical_team(match["team2"]),
            "home_goals": int(ft[0]),
            "away_goals": int(ft[1]),
        })
    return rows


def parse_uefa_txt(text: str, season: str, competition: str) -> list[dict]:
    header_year = None
    head = re.search(r"#\s*Date\s+\w{3}\s+\w{3}\s+\d{1,2}\s+(\d{4})", text)
    if head:
        header_year = int(head.group(1))

    rows: list[dict] = []
    current_date: date | None = None
    current_month: int | None = None
    current_year = header_year
    for line in text.splitlines():
        dm = DATE_HEADER_RE.match(line)
        if dm:
            month = MONTHS[dm.group(1)]
            day = int(dm.group(2))
            explicit_year = int(dm.group(3)) if dm.group(3) else None
            if explicit_year is not None:
                current_year = explicit_year
            elif current_year is None:
                continue
            elif current_month is not None and month < current_month:
                current_year += 1
            try:
                current_date = date(current_year, month, day)
            except ValueError:
                current_date = None
            current_month = month
            continue

        if current_date is None or " v " not in line:
            continue
        mm = MATCH_LINE_RE.match(line)
        if not mm:
            continue
        home = canonical_team(_strip_country(mm.group(1)))
        away = canonical_team(_strip_country(mm.group(3)))
        tail = mm.group(5)
        aet = AET_RE.search(tail)
        if aet:
            hg, ag = int(aet.group(1)), int(aet.group(2))
        else:
            score = SCORE_RE.search(tail)
            if not score:
                continue
            hg, ag = int(score.group(1)), int(score.group(2))
        rows.append({
            "date": str(current_date),
            "competition": competition,
            "season": season,
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
        })
    return rows


# ── build results ────────────────────────────────────────────────────────────

def build_results(seasons: list[str] | None = None, *, refresh: bool = False) -> pd.DataFrame:
    seasons = seasons or HISTORY_SEASONS
    rows: list[dict] = []
    for season in seasons:
        folder = season_folder(season)
        for code, competition in DOMESTIC_COMPS.items():
            url = f"{OPENFOOTBALL_JSON}/{folder}/{code}.json"
            cache = _cache_path(folder, f"{code}.json")
            try:
                text = _fetch(url, cache, refresh=refresh)
            except RuntimeError as exc:
                print(f"WARN: {competition} {folder}: {exc}")
                continue
            rows.extend(parse_domestic_json(text, folder, competition))
        for code, competition in UEFA_COMPS.items():
            url = f"{OPENFOOTBALL_UEFA}/{folder}/{code}.txt"
            cache = _cache_path(folder, f"{code}.txt")
            try:
                text = _fetch(url, cache, refresh=refresh)
            except RuntimeError:
                continue  # competition/season not published
            rows.extend(parse_uefa_txt(text, folder, competition))

    df = pd.DataFrame(rows, columns=[
        "date", "competition", "season", "home_team", "away_team",
        "home_goals", "away_goals",
    ])
    return df.drop_duplicates(subset=["date", "home_team", "away_team"]).sort_values("date").reset_index(drop=True)


def validate_results(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if df.empty:
        return ["european results: empty dataset"], warnings

    parsed = dates.parse_series(df["date"])
    bad_dates = int(parsed.isna().sum())
    if bad_dates:
        errors.append(f"european results: {bad_dates} unparseable dates")
    work = df.assign(match_date=parsed).dropna(subset=["match_date"])

    dupes = work[work.duplicated(subset=["match_date", "home_team", "away_team"], keep=False)]
    if not dupes.empty:
        errors.append(f"european results: {len(dupes)} duplicate match(es)")

    for col in ("home_goals", "away_goals"):
        bad = work[(work[col] < 0) | (work[col] > 15)]
        if not bad.empty:
            errors.append(f"european results: {len(bad)} implausible {col}")
        if work[col].isna().any():
            errors.append(f"european results: missing {col}")

    current_folder = season_folder(HISTORY_SEASONS[0])
    for (competition, season), group in work.groupby(["competition", "season"]):
        if (
            competition in DOMESTIC_COMPS.values()
            and season != current_folder
            and len(group) < EXPECTED_DOMESTIC_MIN
        ):
            warnings.append(
                f"{competition} {season}: only {len(group)} matches "
                f"(expected >= {EXPECTED_DOMESTIC_MIN})"
            )
    return errors, warnings


# ── cross-league Elo ─────────────────────────────────────────────────────────

def build_elo(results: pd.DataFrame, *, k: float = 20.0, home_advantage: float = 65.0):
    """Sequential cross-league Elo; returns (history, current, pre-match map)."""
    ratings: dict[str, float] = {}
    games: dict[str, int] = {}
    history: list[dict] = []
    if results.empty:
        return pd.DataFrame(history), pd.DataFrame(), {}

    work = results.copy()
    work["match_date"] = dates.parse_series(work["date"])
    work = work.dropna(subset=["match_date"]).sort_values(["match_date", "competition"]).reset_index(drop=True)

    for row in work.itertuples(index=False):
        home, away = str(row.home_team), str(row.away_team)
        rh = ratings.get(home, 1500.0)
        ra = ratings.get(away, 1500.0)
        eh = 1.0 / (1.0 + 10 ** ((ra - (rh + home_advantage)) / 400.0))
        gd = int(row.home_goals) - int(row.away_goals)
        if gd > 0:
            sh = 1.0
        elif gd < 0:
            sh = 0.0
        else:
            sh = 0.5
        # Standard Elo: draws still move ratings; goal margin only scales wins.
        margin = 1.0 if gd == 0 else math.log(abs(gd) + 1.0)
        delta = k * margin
        new_h = rh + delta * (sh - eh)
        new_a = ra + delta * ((1.0 - sh) - (1.0 - eh))
        ratings[home], ratings[away] = new_h, new_a
        games[home] = games.get(home, 0) + 1
        games[away] = games.get(away, 0) + 1
        day = str(row.match_date.date())
        history.append({"date": day, "team": home, "elo": new_h, "games": games[home]})
        history.append({"date": day, "team": away, "elo": new_a, "games": games[away]})

    hist = pd.DataFrame(history).sort_values(["date", "team"]).reset_index(drop=True)
    current = pd.DataFrame(
        [{"team": team, "elo": float(ratings[team]), "games": int(games.get(team, 0))} for team in ratings]
    ).sort_values("elo", ascending=False).reset_index(drop=True)
    return hist, current


def _asof_lookup(history: pd.DataFrame):
    """Build a function (team, date) -> (elo, games) using ratings strictly before date."""
    by_team: dict[str, tuple[list[str], list[float], list[int]]] = {}
    for row in history.itertuples(index=False):
        entry = by_team.setdefault(str(row.team), ([], [], []))
        entry[0].append(str(row.date))
        entry[1].append(float(row.elo))
        entry[2].append(int(row.games))

    def lookup(team: str, day: str) -> tuple[float | None, int]:
        entry = by_team.get(str(team))
        if not entry:
            return None, 0
        days, elos, games = entry
        lo, hi = 0, len(days)
        while lo < hi:
            mid = (lo + hi) // 2
            if days[mid] < day:
                lo = mid + 1
            else:
                hi = mid
        if lo == 0:
            return None, 0
        return elos[lo - 1], games[lo - 1]

    return lookup


def _uefa_dates(results: pd.DataFrame) -> dict[str, list[str]]:
    by_team: dict[str, list[str]] = {}
    uefa = results[results["competition"].isin(UEFA_COMPS.values())]
    for row in uefa.itertuples(index=False):
        by_team.setdefault(str(row.home_team), []).append(str(row.date))
        by_team.setdefault(str(row.away_team), []).append(str(row.date))
    return {team: sorted(set(days)) for team, days in by_team.items()}


def _rest_features(team: str, day: str, uefa_days: dict[str, list[str]]) -> tuple[float, float, int, int]:
    days = uefa_days.get(team, [])
    if not days:
        return float("nan"), float("nan"), 0, 0
    prev = [d for d in days if d < day]
    nxt = [d for d in days if d > day]
    since = (date.fromisoformat(day) - date.fromisoformat(prev[-1])).days if prev else float("nan")
    until = (date.fromisoformat(nxt[0]) - date.fromisoformat(day)).days if nxt else float("nan")
    played_midweek = int(not pd.isna(since) and 2 <= since <= 4)
    next_midweek = int(not pd.isna(until) and 2 <= until <= 4)
    return since, until, played_midweek, next_midweek


def build_pl_features(results: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Cross-league Elo + UEFA rest features for every PL match in our history."""
    clean = pd.read_csv(CLEAN_FILE)
    clean["match_date"] = dates.parse_series(clean["Date"])
    clean = clean.dropna(subset=["match_date", "HomeTeam", "AwayTeam"])

    lookup = _asof_lookup(history)
    uefa_days = _uefa_dates(results)
    rows: list[dict] = []
    for row in clean.itertuples(index=False):
        day = str(row.match_date.date())
        home, away = str(row.HomeTeam), str(row.AwayTeam)
        h_elo, h_games = lookup(home, day)
        a_elo, a_games = lookup(away, day)
        h_since, h_until, h_mid, h_next = _rest_features(home, day, uefa_days)
        a_since, a_until, a_mid, a_next = _rest_features(away, day, uefa_days)
        rows.append({
            "date": day,
            "home_team": home,
            "away_team": away,
            "euroelo_home": h_elo,
            "euroelo_away": a_elo,
            "euroelo_diff": None if h_elo is None or a_elo is None else h_elo - a_elo,
            "euroelo_home_games": h_games,
            "euroelo_away_games": a_games,
            "euro_days_since": h_since if not pd.isna(h_since) else None,
            "euro_days_until": h_until if not pd.isna(h_until) else None,
            "euro_played_midweek": h_mid,
            "euro_next_midweek": h_next,
            "euro_away_days_since": None if pd.isna(a_since) else a_since,
            "euro_away_days_until": None if pd.isna(a_until) else a_until,
            "euro_away_played_midweek": a_mid,
            "euro_away_next_midweek": a_next,
        })
    return pd.DataFrame(rows)


# ── external Elo ingestion ──────────────────────────────────────────────────

def import_external_elo(path: str) -> pd.DataFrame:
    """Validate a user-supplied ClubElo CSV and write canonical external_elo.csv.

    Accepted columns (case-insensitive): team/club, elo, optional date.
    Dates make the rows usable for historical features; without dates the file
    is treated as a current snapshot (serving-only).
    """
    raw = pd.read_csv(path)
    columns = {c.lower().strip(): c for c in raw.columns}
    team_col = columns.get("team") or columns.get("club")
    elo_col = columns.get("elo")
    date_col = columns.get("date")
    if not team_col or not elo_col:
        raise SystemExit("external Elo: need columns 'team' (or 'club') and 'elo'")

    work = pd.DataFrame({
        "team": raw[team_col].astype(str).map(clubelo_key).str.strip(),
        "elo": pd.to_numeric(raw[elo_col], errors="coerce"),
    })
    if date_col:
        work["date"] = [dates.iso_date(v) for v in raw[date_col]]
    else:
        work["date"] = str(dates.today_uk())

    bad_elo = work["elo"].isna() | (work["elo"] < 1000) | (work["elo"] > 2600)
    bad_date = work["date"].isna()
    if bad_elo.any():
        print(f"WARN: dropping {int(bad_elo.sum())} row(s) with invalid Elo")
    if bad_date.any():
        print(f"WARN: dropping {int(bad_date.sum())} row(s) with invalid dates")
    work = work[~bad_elo & ~bad_date].drop_duplicates(subset=["team", "date"])
    if work.empty:
        raise SystemExit("external Elo: no usable rows")

    work = work.sort_values(["team", "date"]).reset_index(drop=True)
    work.to_csv(EXTERNAL_ELO_FILE, index=False)
    print(f"OK: {len(work)} canonical rating(s) -> {EXTERNAL_ELO_FILE}")
    print(f"   teams: {work['team'].nunique()} | dated: {date_col is not None}")
    return work


# ── report / CLI ────────────────────────────────────────────────────────────

def report(results: pd.DataFrame, history: pd.DataFrame, features: pd.DataFrame) -> None:
    print("\nEuropean results")
    print(results.groupby(["competition", "season"]).size().unstack(fill_value=0).to_string())
    print("\nElo history teams:", history["team"].nunique(), "| rows:", len(history))
    for team in ["Arsenal", "Man City", "Liverpool", "Chelsea", "Man United", "Newcastle",
                 "Real Madrid", "Barcelona", "Bayern München", "Paris Saint-Germain"]:
        sub = history[history["team"] == team]
        if not sub.empty:
            last = sub.iloc[-1]
            print(f"   {team:<24} elo={float(last['elo']):7.1f} games={int(last['games'])}")
    print("\nPL feature coverage")
    for col in ["euroelo_diff", "euro_days_since", "euro_played_midweek"]:
        print(f"   {col}: {features[col].notna().sum()}/{len(features)} non-null")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="European results + cross-league Elo")
    parser.add_argument("--refresh", action="store_true", help="ignore raw download cache")
    parser.add_argument("--report", action="store_true", help="print coverage report only")
    parser.add_argument("--import-elo", metavar="PATH", help="validate a ClubElo CSV and write external_elo.csv")
    parser.add_argument("--seasons", nargs="*", help="override seasons (default: HISTORY_SEASONS)")
    args = parser.parse_args(argv)

    if args.import_elo:
        import_external_elo(args.import_elo)
        return 0

    results = build_results(args.seasons, refresh=args.refresh)
    errors, warnings = validate_results(results)
    for message in warnings:
        print(f"WARN: {message}")
    if errors:
        for message in errors:
            print(f"ERROR: {message}")
        raise SystemExit("ERROR: European results failed validation - nothing written")

    history, current = build_elo(results)
    features = build_pl_features(results, history)

    results.to_csv(RESULTS_FILE, index=False)
    history.to_csv(ELO_HISTORY_FILE, index=False)
    current.to_csv(ELO_CURRENT_FILE, index=False)
    features.to_csv(FEATURES_FILE, index=False)
    print(f"OK: {len(results):,} matches -> {RESULTS_FILE}")
    print(f"OK: {len(history):,} rating rows -> {ELO_HISTORY_FILE}")
    print(f"OK: {len(features):,} PL feature rows -> {FEATURES_FILE}")

    if args.report:
        report(results, history, features)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
