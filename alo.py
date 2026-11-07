"""alo.py — external European club rankings ("Alo", ClubElo/ClubAlo export).

ALO is intentionally separate from:

* ``elo_*``     — our own Premier League sequential ELO (training feature)
* ``euroelo_*`` — our computed cross-league Elo from European results (candidate)

Alo is a *current* European ranking snapshot supplied manually. It has no
historical coverage, so it is served as reference fields (``alo_home``,
``alo_away``, ``alo_diff``) and never enters ``MODEL_FEATURE_COLS`` or the
market blend.

Usage:
    python alo.py --import C:\\path\\ClubAloRankings.csv
    # expected columns: Club Ranking, Club Name, Club ALO, Country
    # -> alo_rankings.csv (canonical PL names, thousands separators stripped)
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from team_aliases import elo_lookup_key

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ALO_FILE = os.getenv("AUGO_ALO_FILE") or os.path.join(APP_DIR, "alo_rankings.csv")

# ClubElo/ClubAlo short names -> our football-data clean names (PL only).
ALO_TO_CLEAN: dict[str, str] = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Chelsea": "Chelsea",
    "Coventry": "Coventry",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Forest": "Nott'm Forest",
    "Fulham": "Fulham",
    "Hull": "Hull",
    "Ipswich": "Ipswich",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Man City": "Man City",
    "Man United": "Man United",
    "Newcastle": "Newcastle",
    "Nott'm Forest": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",
    "Southampton": "Southampton",
    "Sunderland": "Sunderland",
    "Tottenham": "Tottenham",
    "West Ham": "West Ham",
    "Wolves": "Wolves",
}

_ALO_CACHE: dict[str, tuple[float | None, dict[str, float] | None]] = {}


def alo_key(name: str) -> str:
    """Normalise a ClubAlo name to our clean PL name; pass through others."""
    raw = str(name).strip()
    if raw in ALO_TO_CLEAN:
        return ALO_TO_CLEAN[raw]
    return elo_lookup_key(raw)


def load_alo(path: str | None = None) -> dict[str, float] | None:
    """Load ``team -> alo`` from alo_rankings.csv, cached by mtime."""
    target = path or ALO_FILE
    try:
        mtime = os.path.getmtime(target)
    except OSError:
        return None
    cached = _ALO_CACHE.get(target)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        df = pd.read_csv(target)
        ratings = {
            str(row.team): float(row.alo)
            for row in df.itertuples(index=False)
            if pd.notna(getattr(row, "team", None)) and pd.notna(getattr(row, "alo", None))
        }
    except Exception:
        return None
    _ALO_CACHE[target] = (mtime, ratings)
    return ratings


def _find_column(columns: dict[str, str], *candidates: str) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def import_alo(path: str, *, out: str | None = None) -> pd.DataFrame:
    """Validate a ClubAlo export and write the canonical ``alo_rankings.csv``."""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    columns = {str(c).strip().lower(): c for c in raw.columns}
    name_col = _find_column(columns, "club name", "team", "club")
    alo_col = _find_column(columns, "club alo", "alo", "elo", "rating")
    country_col = _find_column(columns, "country", "country ")
    rank_col = _find_column(columns, "club ranking", "rank")
    if not name_col or not alo_col:
        raise SystemExit("Alo import: need a club/team column and a Club ALO/ALO column")

    alo = pd.to_numeric(
        raw[alo_col].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    work = pd.DataFrame({
        "team": raw[name_col].astype(str).map(alo_key).str.strip(),
        "alo": alo,
    })
    work["source_name"] = raw[name_col].astype(str).str.strip()
    if country_col:
        work["country"] = raw[country_col].astype(str).str.strip()
    else:
        work["country"] = ""
    if rank_col:
        work["rank"] = pd.to_numeric(raw[rank_col], errors="coerce")
    else:
        work["rank"] = pd.Series(range(1, len(work) + 1), dtype=float)

    bad = work["alo"].isna() | (work["alo"] < 1000) | (work["alo"] > 2600)
    if bad.any():
        print(f"WARN: dropping {int(bad.sum())} row(s) with invalid Alo")
    work = work[~bad]
    work = work.drop_duplicates(subset=["team"], keep="first")
    if work.empty:
        raise SystemExit("Alo import: no usable rows")
    work = work.sort_values(["rank", "alo"], ascending=[True, False]).reset_index(drop=True)

    target = out or ALO_FILE
    work.to_csv(target, index=False)
    print(f"OK: {len(work)} Alo ratings -> {target}")
    print(f"   countries: {', '.join(sorted(set(work['country']) - {''})) or 'n/a'}")
    return work


def attach_alo(upcoming: pd.DataFrame, ratings: dict[str, float] | None = None) -> pd.DataFrame:
    """Add serving-only Alo reference columns; never model features."""
    ratings = ratings if ratings is not None else load_alo()
    if not ratings or upcoming.empty:
        return upcoming

    def _rating(team) -> float | None:
        return ratings.get(alo_key(str(team)))

    upcoming = upcoming.copy()
    upcoming["alo_home"] = upcoming["home_team"].map(_rating)
    upcoming["alo_away"] = upcoming["away_team"].map(_rating)
    upcoming["alo_diff"] = upcoming["alo_home"] - upcoming["alo_away"]
    return upcoming


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import ClubAlo European rankings")
    parser.add_argument("--import", dest="import_path", metavar="PATH", required=True)
    parser.add_argument("--out", default=None, help="output CSV (default: alo_rankings.csv)")
    args = parser.parse_args(argv)
    import_alo(args.import_path, out=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
