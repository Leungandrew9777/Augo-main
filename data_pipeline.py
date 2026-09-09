# data_pipeline.py
import pandas as pd
from pathlib import Path
import numpy as np
import json
import time

import requests

class FootballDataLoader:
    # NOTE: no "www." — football-data.co.uk returns 503 with www and works without.
    BASE_URL = "https://football-data.co.uk/mmz4281"
    LEAGUES = {"E0": "Premier League"}

    # Core columns we actually need for the model + odds
    COLUMNS_TO_KEEP = [
        "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR",           # Full-time result & goals
        "HTHG", "HTAG", "HTR",           # Half-time result & goals
        "HS", "AS", "HST", "AST",        # Shots & shots on target
        "HF", "AF", "HC", "AC",          # Fouls & corners
        "HY", "AY", "HR", "AR",          # Cards
        "B365H", "B365D", "B365A",       # Bet365 1X2 odds
        "B365>2.5", "B365<2.5",          # Bet365 over/under 2.5 goals
        "HxG", "AxG",                    # Football-Data xG (when available)
    ]

    def __init__(self, seasons: list[str]):
        self.seasons = seasons   # e.g. ["2526", "2425", "2324", ...]

    def load_season(self, league: str, season: str) -> pd.DataFrame:
        """Load one season/league CSV with robust parsing."""
        url = f"{self.BASE_URL}/{season}/{league}.csv"
        try:
            # This combination works for all seasons including 2526
            df = pd.read_csv(
                url,
                encoding="ISO-8859-1",      # ← This fixes most parsing issues
                on_bad_lines="skip",
                low_memory=False
            )
            # Keep only columns that actually exist
            available_cols = [c for c in self.COLUMNS_TO_KEEP if c in df.columns]
            df = df[available_cols].copy()

            df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
            df["League"] = self.LEAGUES.get(league, league)
            df["Season"] = season
            print(f"✓ {self.LEAGUES.get(league)} {season}: {len(df)} matches")
            return df
        except Exception as e:
            print(f"⚠️  Failed to load {league}/{season}: {e}")
            return pd.DataFrame()

    def load_all(self) -> pd.DataFrame:
        frames = []
        for league in self.LEAGUES:
            for season in self.seasons:
                df = self.load_season(league, season)
                if not df.empty:
                    frames.append(df)
        result = pd.concat(frames, ignore_index=True)
        print(f"\n✅ Total matches loaded: {len(result):,}")
        return result


class DataCleaner:
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        numeric_cols = [
            "FTHG", "FTAG", "HTHG", "HTAG", "HS", "AS", "HST", "AST",
            "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
            "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5",
            "HxG", "AxG",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Encode result for model: H=2, D=1, A=0
        result_map = {"H": 2, "D": 1, "A": 0}
        df["Result"] = df["FTR"].map(result_map)
        df = df.dropna(subset=["Result"])
        df["Result"] = df["Result"].astype(int)
        return df      


# ── Real xG (Understat) ────────────────────────────────────────────────────────
# Football-Data does not publish xG, so per-match expected goals are pulled from
# Understat (https://understat.com) and merged onto the cleaned frame by
# (date, home_team, away_team). Where Understat has no entry (older seasons or
# mapping misses) the shot-based proxy in feature_engineering.py is used instead.

UNDERSTAT_BASE = "https://understat.com"
UNDERSTAT_LEAGUE_PAGE = f"{UNDERSTAT_BASE}/league/EPL/{{year}}"
UNDERSTAT_DATA_URL = f"{UNDERSTAT_BASE}/main/getLeagueData/EPL/{{year}}"

# Understat team titles -> football-data.co.uk short names used in this project
UNDERSTAT_TO_SHORT: dict[str, str] = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest": "Nott'm Forest",
    "West Bromwich Albion": "West Brom",
}

UNDERSTAT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _understat_short(title: str) -> str:
    s = str(title).strip()
    return UNDERSTAT_TO_SHORT.get(s, s)


def _understat_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": UNDERSTAT_USER_AGENT,
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json, text/javascript, */*; q=0.01",
    })
    return s


def fetch_understat_xg(seasons: list[str], sleep: float = 0.4) -> pd.DataFrame:
    """Fetch per-match xG from Understat for each Football-Data season code.

    ``seasons`` are Football-Data codes like ``"2526"``; Understat uses the
    season start year (2025 for 2025/26). Returns a frame with columns
    ``date, home_team, away_team, home_xg, away_xg`` (date is ISO ``YYYY-MM-DD``).
    """
    sess = _understat_session()
    rows: list[dict] = []
    for season in seasons:
        year = int(str(int(season))[:2]) + 2000
        if year < 2014 or year > 2026:
            print(f"   Understat has no xG for season {season} — skipping.")
            continue
        try:
            # Visiting the league page establishes the session cookie the API needs.
            sess.get(UNDERSTAT_LEAGUE_PAGE.format(year=year), timeout=25)
            resp = sess.get(UNDERSTAT_DATA_URL.format(year=year), timeout=25)
            resp.raise_for_status()
            data = json.loads(resp.text)
        except Exception as exc:
            print(f"⚠️  Understat {year} fetch failed: {exc}")
            time.sleep(sleep)
            continue
        matches = data.get("dates", [])
        cnt = 0
        for m in matches:
            if not m.get("isResult"):
                continue
            try:
                h_xg = float(m["xG"]["h"])
                a_xg = float(m["xG"]["a"])
            except (TypeError, ValueError, KeyError):
                continue
            day = str(m.get("datetime", ""))[:10]
            if not day:
                continue
            rows.append({
                "date": day,
                "home_team": _understat_short(m["h"]["title"]),
                "away_team": _understat_short(m["a"]["title"]),
                "home_xg": h_xg,
                "away_xg": a_xg,
            })
            cnt += 1
        print(f"✓ Understat xG {year}: {cnt} matches")
        time.sleep(sleep)
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_xg", "away_xg"])


def merge_understat_xg(clean_df: pd.DataFrame, xg_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join real xG onto the cleaned frame; missing rows stay NaN."""
    out = clean_df.copy()
    out["home_xg"] = np.nan
    out["away_xg"] = np.nan
    if xg_df is None or xg_df.empty:
        return out
    xg = xg_df.copy()
    xg["date"] = pd.to_datetime(xg["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    xg = xg.drop_duplicates(subset=["date", "home_team", "away_team"])
    lookup = xg.set_index(["date", "home_team", "away_team"])
    keys = list(zip(
        pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d"),
        out["HomeTeam"].astype(str).str.strip(),
        out["AwayTeam"].astype(str).str.strip(),
    ))
    for i, key in enumerate(keys):
        if key in lookup.index:
            row = lookup.loc[key]
            out.at[i, "home_xg"] = float(row["home_xg"])
            out.at[i, "away_xg"] = float(row["away_xg"])
    # Fall back to football-data's own xG columns (HxG/AxG) where Understat misses
    if "HxG" in out.columns:
        out["home_xg"] = out["home_xg"].fillna(pd.to_numeric(out["HxG"], errors="coerce"))
    if "AxG" in out.columns:
        out["away_xg"] = out["away_xg"].fillna(pd.to_numeric(out["AxG"], errors="coerce"))
    matched = out["home_xg"].notna().sum()
    print(f"   xG matched {matched}/{len(out)} matches.")
    return out


# ====================== RUN THIS ======================
if __name__ == "__main__":
    # Include the current season (2627) + all previous you want
    loader = FootballDataLoader(
        seasons=["2627", "2526", "2425", "2324", "2223", "2122", "2021", "1920", "1819", "1718", "1617", "1516", "1415", "1314"]
    )
    raw_data = loader.load_all()

    clean_data = DataCleaner.clean(raw_data)
    clean_data.to_csv("premier_league_historical_clean.csv", index=False)

    print(f"✅ Saved {len(clean_data):,} cleaned matches → premier_league_historical_clean.csv")

    # Attach real xG from Understat where available
    try:
        print("Fetching real xG from Understat …")
        xg = fetch_understat_xg(loader.seasons)
        clean_data = merge_understat_xg(clean_data, xg)
        clean_data.to_csv("premier_league_historical_clean.csv", index=False)
        print(f"✅ Saved {len(clean_data):,} cleaned matches with xG → premier_league_historical_clean.csv")
    except Exception as exc:
        print(f"⚠️  Understat xG merge skipped: {exc}")
    print("Now run the feature engineering + training steps below.")
