# feature_engineering.py
import json
import os
from datetime import datetime

import pandas as pd
import numpy as np

from dates import parse_series

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_FILE = os.path.join(APP_DIR, "premier_league_historical_clean.csv")
FEATURES_FILE = os.path.join(APP_DIR, "premier_league_with_elo_best.csv")
STATE_FILE = os.path.join(APP_DIR, "team_state.json")

WINDOW = 5
STATE_DUMMY_TEAM = "__augo_state__"

# League-average ELO used to normalise strength-of-schedule adjustments.
LEAGUE_AVG_ELO = 1500.0
STRENGTH_CLAMP = (0.5, 2.0)


class FeatureEngineer:
    def __init__(self, window: int = 5):
        self.window = window

    @staticmethod
    def _attack_factor(opp_elo: pd.Series) -> pd.Series:
        """Weight for attacking stats: bigger reward against stronger opposition."""
        return np.clip(opp_elo / LEAGUE_AVG_ELO, *STRENGTH_CLAMP)

    @staticmethod
    def _defense_factor(opp_elo: pd.Series) -> pd.Series:
        """Weight for defensive stats: conceding vs strong sides counts less."""
        return np.clip(LEAGUE_AVG_ELO / opp_elo, *STRENGTH_CLAMP)

    def compute_team_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        base_cols = [
            "Date", "Team", "GF", "GA", "Shots", "ShotsAgainst", "SoT", "SoTAgainst",
            "Corners", "CornersAgainst", "Fouls", "FoulsAgainst",
            "OwnXG", "OppXG", "OppElo",
        ]
        # Home-venue records: opponent is the away side; its xG is what the
        # home defence faced, and the away rating is the opponent strength.
        home = df[["Date", "HomeTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST",
                   "HC", "AC", "HF", "AF",
                   "home_xg", "away_xg", "elo_away_rating"]].copy()
        home.columns = base_cols
        home["IsHome"] = 1
        # Away-venue records mirrored.
        away = df[["Date", "AwayTeam", "FTAG", "FTHG", "AS", "HS", "AST", "HST",
                   "AC", "HC", "AF", "HF",
                   "away_xg", "home_xg", "elo_home_rating"]].copy()
        away.columns = base_cols
        away["IsHome"] = 0

        all_records = pd.concat([home, away]).sort_values("Date").reset_index(drop=True)

        # xG: real Understat xG when available, otherwise the shot-based proxy.
        proxy_xg = all_records["SoT"] * 0.30 + (all_records["Shots"] - all_records["SoT"]) * 0.03
        proxy_xga = (
            all_records["SoTAgainst"] * 0.30
            + (all_records["ShotsAgainst"] - all_records["SoTAgainst"]) * 0.03
        )
        if all_records["OwnXG"].notna().any():
            all_records["xG"] = all_records["OwnXG"].fillna(proxy_xg)
        else:
            all_records["xG"] = proxy_xg
        if all_records["OppXG"].notna().any():
            all_records["xGA"] = all_records["OppXG"].fillna(proxy_xga)
        else:
            all_records["xGA"] = proxy_xga
        all_records["xG_overperf"] = all_records["GF"] - all_records["xG"]
        all_records["Points"] = np.where(
            all_records["GF"] > all_records["GA"], 3,
            np.where(all_records["GF"] == all_records["GA"], 1, 0),
        )

        attack_cols = ["GF", "Shots", "SoT", "Corners", "Fouls", "xG", "xG_overperf", "Points"]
        defense_cols = ["GA", "ShotsAgainst", "SoTAgainst", "CornersAgainst", "FoulsAgainst", "xGA"]
        stat_cols = [
            "GF", "GA", "Shots", "ShotsAgainst", "SoT", "SoTAgainst",
            "Corners", "CornersAgainst", "Fouls", "FoulsAgainst",
            "xG", "xGA", "xG_overperf",
        ]

        def _venue_stats(venue: pd.DataFrame) -> pd.DataFrame:
            v = venue.copy()
            atk = self._attack_factor(v["OppElo"])
            dfn = self._defense_factor(v["OppElo"])
            for col in attack_cols:
                v[col] = v[col] * atk
            for col in defense_cols:
                v[col] = v[col] * dfn
            for col in stat_cols:
                v[f"avg_{col}"] = v[col].shift(1).rolling(self.window, min_periods=3).mean()
            v["Form"] = v["Points"].shift(1).rolling(self.window, min_periods=3).mean()
            return v

        frames = [
            _venue_stats(venue)
            for _, venue in all_records.groupby(["Team", "IsHome"], sort=False)
            if not venue.empty
        ]
        if not frames:
            return all_records
        return pd.concat(frames).sort_values("Date").reset_index(drop=True)

    def build_match_features(self, df: pd.DataFrame) -> pd.DataFrame:
        team_stats = self.compute_team_stats(df)
        stat_features = [c for c in team_stats.columns if c.startswith("avg_")] + ["Form"]

        home_stats = team_stats.loc[
            team_stats["IsHome"] == 1, ["Team", "Date"] + stat_features
        ].rename(columns={"Team": "HomeTeam", **{c: f"home_{c}" for c in stat_features}})
        away_stats = team_stats.loc[
            team_stats["IsHome"] == 0, ["Team", "Date"] + stat_features
        ].rename(columns={"Team": "AwayTeam", **{c: f"away_{c}" for c in stat_features}})

        base = df.reset_index(drop=True)
        feature_frame = base[["Date", "HomeTeam", "AwayTeam"]].merge(
            home_stats, on=["Date", "HomeTeam"], how="left",
        ).merge(
            away_stats, on=["Date", "AwayTeam"], how="left",
        )
        for feat in stat_features:
            feature_frame[f"diff_{feat}"] = (
                feature_frame[f"home_{feat}"] - feature_frame[f"away_{feat}"]
            )
        feature_frame = feature_frame.drop(columns=["Date", "HomeTeam", "AwayTeam"])
        result = pd.concat([base, feature_frame], axis=1)
        # Only drop rows without a computed ELO diff. Rolling venue stats can be
        # NaN for new teams / early-season games (not enough history), and the
        # trainer median-fills them; dropping on all features would silently
        # remove brand-new promoted teams entirely.
        core = [c for c in ("elo_diff",) if c in result.columns]
        return result.dropna(subset=core)


class FootballELO:
    def __init__(self, k: int = 32, home_advantage: int = 65):
        self.k = k
        self.home_advantage = home_advantage
        self.ratings: dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        return self.ratings.setdefault(team, 1500.0)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def margin_multiplier(self, goal_diff: int) -> float:
        return np.log(abs(goal_diff) + 1) * (2.2 / 2.2)   # simplified FiveThirtyEight style

    def update(self, home: str, away: str, home_goals: int, away_goals: int):
        r_home = self.get_rating(home) + self.home_advantage
        r_away = self.get_rating(away)
        e_home = self.expected_score(r_home, r_away)

        if home_goals > away_goals:
            s_home, s_away = 1.0, 0.0
        elif home_goals < away_goals:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        m = self.margin_multiplier(home_goals - away_goals)
        self.ratings[home] += self.k * m * (s_home - e_home)
        self.ratings[away] += self.k * m * (s_away - (1 - e_home))

    def compute_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("Date").copy()
        for _, row in df.iterrows():
            r_home = self.get_rating(row["HomeTeam"])
            r_away = self.get_rating(row["AwayTeam"])
            df.at[row.name, "elo_home_before"] = r_home + self.home_advantage
            df.at[row.name, "elo_away_before"] = r_away
            df.at[row.name, "elo_home_rating"] = r_home
            df.at[row.name, "elo_away_rating"] = r_away
            df.at[row.name, "elo_diff"] = (r_home + self.home_advantage) - r_away
            self.update(row["HomeTeam"], row["AwayTeam"], int(row["FTHG"]), int(row["FTAG"]))
        return df


def add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    if all(col in df.columns for col in ["B365H", "B365D", "B365A"]):
        df["odds_prob_H"] = 1 / df["B365H"]
        df["odds_prob_D"] = 1 / df["B365D"]
        df["odds_prob_A"] = 1 / df["B365A"]
        total = df["odds_prob_H"] + df["odds_prob_D"] + df["odds_prob_A"]
        df["norm_prob_H"] = df["odds_prob_H"] / total
        df["norm_prob_D"] = df["odds_prob_D"] / total
        df["norm_prob_A"] = df["odds_prob_A"] / total
        df["odds_spread"] = df["norm_prob_H"] - df["norm_prob_A"]
    return df


def add_fatigue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Single chronological pass: rest days per team, advantage flag."""
    df = df.copy()
    df["Date"] = parse_series(df["Date"])
    df = df.sort_values("Date")
    last_match: dict[str, pd.Timestamp] = {}
    home_rest = []
    away_rest = []

    for _, row in df.iterrows():
        h, a, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
        home_rest.append((date - last_match[h]).days if h in last_match else np.nan)
        away_rest.append((date - last_match[a]).days if a in last_match else np.nan)
        last_match[h] = date
        last_match[a] = date

    # Clip: multi-season gaps (promoted teams) are just "no congestion"; anything
    # over 30 days carries no additional signal and would skew linear models.
    df["home_rest_days"] = np.clip(home_rest, 0, 30)
    df["away_rest_days"] = np.clip(away_rest, 0, 30)
    df["rest_advantage"] = df["home_rest_days"] - df["away_rest_days"]
    df["home_fatigued"] = (df["home_rest_days"] <= 3).astype(int)
    df["away_fatigued"] = (df["away_rest_days"] <= 3).astype(int)
    return df


def add_h2h_features(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """For each match, look back at the last *n* meetings between the same two
    teams (either direction) and compute stats from the home team's perspective.

    Vectorised: matches are keyed by the unordered pair of teams, then a shifted
    rolling window per pair reproduces the original per-row ``tail(n)`` scan.
    """
    df = df.copy()
    df["Date"] = parse_series(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    home = df["HomeTeam"].astype(str)
    away = df["AwayTeam"].astype(str)
    first_is_home = home.str.casefold() <= away.str.casefold()
    canon_home = home.where(first_is_home, away)
    canon_away = away.where(first_is_home, home)

    hg = pd.to_numeric(df["FTHG"], errors="coerce")
    ag = pd.to_numeric(df["FTAG"], errors="coerce")
    canon_hg = hg.where(first_is_home, ag)
    canon_ag = ag.where(first_is_home, hg)

    work = pd.DataFrame({
        "pair": canon_home + "|" + canon_away,
        "canon_win": (canon_hg > canon_ag).astype(float),
        "draw": (canon_hg == canon_ag).astype(float),
        "total": (hg + ag).astype(float),
    })
    grouped = work.groupby("pair", sort=False)

    def _prior_mean(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(n, min_periods=1).mean()

    win_roll = grouped["canon_win"].transform(_prior_mean)
    draw_roll = grouped["draw"].transform(_prior_mean)
    total_roll = grouped["total"].transform(_prior_mean)

    df["h2h_home_wins"] = win_roll.where(first_is_home, 1.0 - win_roll - draw_roll)
    df["h2h_draws"] = draw_roll
    df["h2h_total_goals_avg"] = total_roll
    return df


def build_team_state(df: pd.DataFrame, elo: "FootballELO", *, window: int = WINDOW) -> dict:
    """Snapshot every team as of AFTER its last completed match.

    Training rows hold rolling stats over the five matches *before* that match
    (``shift(1).rolling``), so serving must shift the same window one match
    forward. We append one synthetic "next match" row per team and venue and
    let :meth:`FeatureEngineer.compute_team_stats` fill its rolling values; the
    synthetic row's own (NaN) stats never enter the average because the window
    is shifted. The result is the exact feature vector a real next fixture
    would receive, plus the post-history ELO ratings.
    """
    generated_at = datetime.now().isoformat(timespec="seconds")
    if df.empty:
        return {
            "generated_at": generated_at, "window": int(window),
            "home_advantage": float(elo.home_advantage),
            "last_match_date": None, "ratings": {}, "teams": {},
        }

    work = df.copy()
    work["Date"] = parse_series(work["Date"])
    work = work.dropna(subset=["Date"])
    for col in ("home_xg", "away_xg"):
        if col not in work.columns:
            work[col] = np.nan

    teams = sorted({str(t) for t in pd.concat([work["HomeTeam"], work["AwayTeam"]]).unique()})
    ratings = {team: float(elo.get_rating(team)) for team in teams}
    state_date = work["Date"].max() + pd.Timedelta(days=1)

    synthetic: list[dict] = []
    for team in teams:
        base = {
            "Date": state_date, "FTHG": np.nan, "FTAG": np.nan,
            "HS": np.nan, "AS": np.nan, "HST": np.nan, "AST": np.nan,
            "HC": np.nan, "AC": np.nan, "HF": np.nan, "AF": np.nan,
            "home_xg": np.nan, "away_xg": np.nan,
        }
        synthetic.append({
            **base, "HomeTeam": team, "AwayTeam": STATE_DUMMY_TEAM,
            "elo_home_rating": ratings[team], "elo_away_rating": 1500.0,
        })
        synthetic.append({
            **base, "HomeTeam": STATE_DUMMY_TEAM, "AwayTeam": team,
            "elo_home_rating": 1500.0, "elo_away_rating": ratings[team],
        })

    combined = pd.concat([work, pd.DataFrame(synthetic)], ignore_index=True)
    stats = FeatureEngineer(window=window).compute_team_stats(combined)
    suffix_cols = [c for c in stats.columns if c.startswith("avg_")] + ["Form"]
    state_rows = stats[(stats["Team"].isin(teams)) & (stats["Date"] == state_date)]

    team_state: dict[str, dict[str, dict[str, float | None]]] = {}
    for _, row in state_rows.iterrows():
        venue = "home" if int(row["IsHome"]) == 1 else "away"
        team_state.setdefault(str(row["Team"]), {})[venue] = {
            col: (None if pd.isna(row[col]) else float(row[col])) for col in suffix_cols
        }

    return {
        "generated_at": generated_at,
        "window": int(window),
        "home_advantage": float(elo.home_advantage),
        "last_match_date": str(work["Date"].max().date()),
        "ratings": ratings,
        "teams": team_state,
    }


# ====================== RUN THIS ======================
if __name__ == "__main__":
    print("Loading cleaned data...")
    df = pd.read_csv(CLEAN_FILE)
    df["Date"] = parse_series(df["Date"])
    df = df.dropna(subset=["Date"])

    print("Computing ELO ratings (needed for strength-of-schedule)...")
    elo = FootballELO(k=32, home_advantage=65)
    df = elo.compute_elo_features(df)

    print("Building home/away-split, strength-adjusted rolling stats...")
    engineer = FeatureEngineer(window=WINDOW)
    featured = engineer.build_match_features(df)

    print("Adding bookmaker odds features...")
    featured = add_odds_features(featured)
    print("Adding H2H features (slow, O(n^2))...")
    featured = add_h2h_features(featured, n=5)
    featured.to_csv(FEATURES_FILE, index=False)
    print(f"OK: STEP 2 COMPLETE - Saved {len(featured):,} matches -> {FEATURES_FILE}")

    print("Snapshotting team state as of the last completed match...")
    state = build_team_state(df, elo, window=WINDOW)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    print(f"OK: team state -> {STATE_FILE} "
          f"(last match {state['last_match_date']}, {len(state['teams'])} teams, "
          f"{len(state['ratings'])} ratings)")
