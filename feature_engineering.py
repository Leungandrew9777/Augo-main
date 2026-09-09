# feature_engineering.py
import pandas as pd
import numpy as np

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

        frames = []
        for team in all_records["Team"].unique():
            team_data = all_records[all_records["Team"] == team]
            for is_home in (1, 0):
                venue = team_data[team_data["IsHome"] == is_home]
                if venue.empty:
                    continue
                frames.append(_venue_stats(venue))
        return pd.concat(frames).sort_values("Date").reset_index(drop=True)

    def build_match_features(self, df: pd.DataFrame) -> pd.DataFrame:
        team_stats = self.compute_team_stats(df)
        stat_features = [c for c in team_stats.columns if c.startswith("avg_")] + ["Form"]

        features_list = []
        for idx, match in df.iterrows():
            home = match["HomeTeam"]
            away = match["AwayTeam"]
            date = match["Date"]

            home_stats = team_stats[(team_stats["Team"] == home) & (team_stats["Date"] == date) & (team_stats["IsHome"] == 1)]
            away_stats = team_stats[(team_stats["Team"] == away) & (team_stats["Date"] == date) & (team_stats["IsHome"] == 0)]

            if home_stats.empty or away_stats.empty:
                continue

            row = {"match_idx": idx}
            for feat in stat_features:
                h_val = home_stats[feat].values[0]
                a_val = away_stats[feat].values[0]
                row[f"home_{feat}"] = h_val
                row[f"away_{feat}"] = a_val
                row[f"diff_{feat}"] = h_val - a_val
            features_list.append(row)

        features_df = pd.DataFrame(features_list).set_index("match_idx")
        # Only drop rows without a computed ELO diff. Rolling venue stats can be
        # NaN for new teams / early-season games (not enough history), and the
        # trainer median-fills them; dropping on all features would silently
        # remove brand-new promoted teams entirely.
        core = [c for c in ("elo_diff",) if c in features_df.columns]
        return df.join(features_df, how="inner").dropna(subset=core)


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
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True, format="mixed")
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

    df["home_rest_days"] = home_rest
    df["away_rest_days"] = away_rest
    df["rest_advantage"] = df["home_rest_days"] - df["away_rest_days"]
    df["home_fatigued"] = (df["home_rest_days"] <= 3).astype(int)
    df["away_fatigued"] = (df["away_rest_days"] <= 3).astype(int)
    return df


def add_h2h_features(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """For each match, look back at the last *n* meetings between the same two
    teams (either direction) and compute stats from the home team's perspective."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True, format="mixed")
    df = df.sort_values("Date")
    h2h_wins = []
    h2h_draws = []
    h2h_avg_goals = []

    for _, row in df.iterrows():
        h, a, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
        prior = df[
            (df["Date"] < date)
            & (
                ((df["HomeTeam"] == h) & (df["AwayTeam"] == a))
                | ((df["HomeTeam"] == a) & (df["AwayTeam"] == h))
            )
        ].tail(n)

        if prior.empty:
            h2h_wins.append(np.nan)
            h2h_draws.append(np.nan)
            h2h_avg_goals.append(np.nan)
            continue

        wins = 0
        draws = 0
        total_goals = 0
        for _, p in prior.iterrows():
            gh = int(p["FTHG"])
            ga = int(p["FTAG"])
            total_goals += gh + ga
            if p["HomeTeam"] == h:
                wins += gh > ga
                draws += gh == ga
            else:
                wins += ga > gh
                draws += gh == ga
        h2h_wins.append(wins / len(prior))
        h2h_draws.append(draws / len(prior))
        h2h_avg_goals.append(total_goals / len(prior))

    df["h2h_home_wins"] = h2h_wins
    df["h2h_draws"] = h2h_draws
    df["h2h_total_goals_avg"] = h2h_avg_goals
    return df


# ====================== RUN THIS ======================
if __name__ == "__main__":
    print("Loading cleaned data...")
    df = pd.read_csv("premier_league_historical_clean.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True, format="mixed")
    df = df.dropna(subset=["Date"])
    print("Computing ELO ratings (needed for strength-of-schedule)...")
    elo = FootballELO(k=32, home_advantage=65)
    df = elo.compute_elo_features(df)
    print("Building home/away-split, strength-adjusted rolling stats...")
    engineer = FeatureEngineer(window=5)
    featured = engineer.build_match_features(df)
    print("Adding bookmaker odds features...")
    featured = add_odds_features(featured)
    print("Adding H2H features (slow, O(n^2))...")
    featured = add_h2h_features(featured, n=5)
    featured.to_csv("premier_league_with_elo_best.csv", index=False)
    print(f"OK: STEP 2 COMPLETE - Saved {len(featured):,} matches -> premier_league_with_elo_best.csv")
