"""
features.py — single source of truth for match-feature construction.

Training (train_ensemble.py) and every inference path (run_pipeline.py,
app.py) build features through this module so the feature set can never
drift apart between the fitted model and live predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from team_aliases import elo_lookup_key

# Canonical feature list used to train the ensemble AND as the inference
# fallback when a model does not expose ``feature_names_in_``.
MODEL_FEATURE_COLS: list[str] = [
    "elo_diff",
    "home_Form", "away_Form",                    # avg points (form)
    "diff_Form",
    "home_avg_GF", "away_avg_GF",
    "home_avg_GA", "away_avg_GA",
    "home_avg_SoT", "away_avg_SoT",
    "home_avg_SoTAgainst", "away_avg_SoTAgainst",
    "home_avg_Shots", "away_avg_Shots",
    "home_avg_ShotsAgainst", "away_avg_ShotsAgainst",
    "home_avg_xG", "away_avg_xG",
    "home_avg_xGA", "away_avg_xGA",
    "home_avg_xG_overperf", "away_avg_xG_overperf",
    "diff_avg_GF", "diff_avg_GA", "diff_avg_SoT", "diff_avg_Shots",
    "diff_avg_SoTAgainst", "diff_avg_ShotsAgainst",
    "diff_avg_xG", "diff_avg_xGA", "diff_avg_xG_overperf",
    "h2h_home_wins", "h2h_draws", "h2h_total_goals_avg",
]


def normalize_elo_columns(df_elo: pd.DataFrame) -> pd.DataFrame:
    """Normalise Football-Data CamelCase columns to snake_case, if needed."""
    rename_map: dict[str, str] = {}
    if "Date" in df_elo.columns and "date" not in df_elo.columns:
        rename_map["Date"] = "date"
    if "HomeTeam" in df_elo.columns and "home_team" not in df_elo.columns:
        rename_map["HomeTeam"] = "home_team"
    if "AwayTeam" in df_elo.columns and "away_team" not in df_elo.columns:
        rename_map["AwayTeam"] = "away_team"
    if "FTR" in df_elo.columns and "result" not in df_elo.columns:
        rename_map["FTR"] = "result"
    if rename_map:
        df_elo = df_elo.rename(columns=rename_map)

    if "result" in df_elo.columns:
        vals = set(pd.Series(df_elo["result"]).dropna().astype(str).unique().tolist())
        if vals.issubset({"0", "1", "2"}):
            df_elo["result"] = df_elo["result"].map(
                {2: "H", 1: "D", 0: "A", "2": "H", "1": "D", "0": "A"}
            )
    return df_elo


def patch_model_runtime_compat(model):
    """Patch pickled LR estimators for sklearn cross-version compatibility."""
    from sklearn.linear_model import LogisticRegression

    for est in getattr(model, "estimators_", []):
        if hasattr(est, "named_steps") and "model" in est.named_steps:
            inner = est.named_steps["model"]
            if isinstance(inner, LogisticRegression) and not hasattr(inner, "multi_class"):
                setattr(inner, "multi_class", "auto")


def load_model_and_elo(model_path: str, elo_path: str, *, patch: bool = True):
    """Load the fitted ensemble and the ELO feature history together."""
    import joblib

    from sklearn.linear_model import LogisticRegression

    model = joblib.load(model_path)
    if patch:
        patch_model_runtime_compat(model)

    df_elo = pd.read_csv(elo_path)
    df_elo = normalize_elo_columns(df_elo)
    df_elo["date"] = pd.to_datetime(df_elo["date"])
    return model, df_elo


def compute_current_elo(
    upcoming: pd.DataFrame,
    df_elo: pd.DataFrame,
    elo_key=elo_lookup_key,
) -> pd.DataFrame:
    """Attach the latest pre-match ELO of each side and the ELO difference."""
    latest_elo: dict[str, float] = {}
    for team in pd.concat([df_elo["home_team"], df_elo["away_team"]]).unique():
        m = df_elo[(df_elo["home_team"] == team) | (df_elo["away_team"] == team)]
        if len(m) > 0:
            last = m.sort_values("date").iloc[-1]
            latest_elo[team] = (
                last["elo_home_before"] if last["home_team"] == team
                else last["elo_away_before"]
            )
        else:
            latest_elo[team] = 1500.0
    upcoming["elo_home"] = upcoming["home_team"].map(
        lambda t: latest_elo.get(elo_key(str(t)), 1500.0)
    )
    upcoming["elo_away"] = upcoming["away_team"].map(
        lambda t: latest_elo.get(elo_key(str(t)), 1500.0)
    )
    upcoming["elo_diff"] = upcoming["elo_home"] - upcoming["elo_away"]
    return upcoming


def latest_team_feature(
    df_elo: pd.DataFrame, team_col: str, team_name: str, feature_col: str,
) -> float | None:
    if feature_col not in df_elo.columns:
        return None
    series = pd.to_numeric(
        df_elo.loc[df_elo[team_col] == team_name, feature_col], errors="coerce",
    ).dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def detect_model_features(model) -> list[str]:
    """Extract expected feature names from a fitted model/ensemble."""
    if hasattr(model, "feature_names_in_"):
        return [str(c) for c in list(getattr(model, "feature_names_in_", []))]
    if hasattr(model, "estimators_") and len(getattr(model, "estimators_", [])) > 0:
        first_est = model.estimators_[0]
        if hasattr(first_est, "feature_names_in_"):
            return [str(c) for c in list(getattr(first_est, "feature_names_in_", []))]
        if hasattr(first_est, "named_steps") and "scaler" in first_est.named_steps:
            scaler = first_est.named_steps["scaler"]
            if hasattr(scaler, "feature_names_in_"):
                return [str(c) for c in list(getattr(scaler, "feature_names_in_", []))]
    return []


def ensure_model_features(
    upcoming: pd.DataFrame,
    df_elo: pd.DataFrame,
    expected_cols: list[str],
) -> pd.DataFrame:
    """Fill every column the model expects, per fixture, from the ELO history.

    Home/away features come from the most recent matching venue row (so the
    home-split / away-split stats built by feature_engineering.py are used),
    ``diff_*`` are derived, and H2H features are computed per fixture from
    prior meetings (not a global median).
    """
    if not expected_cols:
        return upcoming

    medians: dict[str, float] = {}
    for col in expected_cols:
        if col in df_elo.columns:
            s = pd.to_numeric(df_elo[col], errors="coerce").dropna()
            if not s.empty:
                medians[col] = float(s.median())

    def _fill_home(col: str):
        fallback = medians.get(col, 0.0)
        upcoming[col] = upcoming["home_team"].map(
            lambda t: latest_team_feature(df_elo, "home_team", elo_lookup_key(str(t)), col)
        )
        upcoming[col] = pd.to_numeric(upcoming[col], errors="coerce").fillna(fallback)

    def _fill_away(col: str):
        fallback = medians.get(col, 0.0)
        upcoming[col] = upcoming["away_team"].map(
            lambda t: latest_team_feature(df_elo, "away_team", elo_lookup_key(str(t)), col)
        )
        upcoming[col] = pd.to_numeric(upcoming[col], errors="coerce").fillna(fallback)

    if "date" in upcoming.columns and any(c in expected_cols for c in (
        "h2h_home_wins", "h2h_draws", "h2h_total_goals_avg",
    )):
        def _h2h_features(row) -> pd.Series:
            home = elo_lookup_key(str(row["home_team"]))
            away = elo_lookup_key(str(row["away_team"]))
            date = pd.to_datetime(row["date"], errors="coerce")
            if pd.isna(date):
                return pd.Series({
                    "h2h_home_wins": pd.NA, "h2h_draws": pd.NA, "h2h_total_goals_avg": pd.NA,
                })
            prior = df_elo[
                (df_elo["date"] < date)
                & (
                    ((df_elo["home_team"] == home) & (df_elo["away_team"] == away))
                    | ((df_elo["home_team"] == away) & (df_elo["away_team"] == home))
                )
            ].tail(5)
            if prior.empty:
                return pd.Series({
                    "h2h_home_wins": pd.NA, "h2h_draws": pd.NA, "h2h_total_goals_avg": pd.NA,
                })
            wins = 0
            draws = 0
            total_goals = 0
            for _, p in prior.iterrows():
                gh = int(p["FTHG"])
                ga = int(p["FTAG"])
                total_goals += gh + ga
                if p["home_team"] == home:
                    wins += gh > ga
                else:
                    wins += ga > gh
                draws += gh == ga
            return pd.Series({
                "h2h_home_wins": wins / len(prior),
                "h2h_draws": draws / len(prior),
                "h2h_total_goals_avg": total_goals / len(prior),
            })

        h2h = upcoming.apply(_h2h_features, axis=1)
        for col in h2h.columns:
            upcoming[col] = h2h[col]

    for col in expected_cols:
        if col in upcoming.columns:
            upcoming[col] = pd.to_numeric(upcoming[col], errors="coerce").fillna(medians.get(col, 0.0))
            continue
        if col.startswith("home_"):
            _fill_home(col)
        elif col.startswith("away_"):
            _fill_away(col)
        elif col.startswith("diff_"):
            suffix = col[len("diff_"):]
            home_col = f"home_{suffix}"
            away_col = f"away_{suffix}"
            if home_col not in upcoming.columns:
                _fill_home(home_col)
            if away_col not in upcoming.columns:
                _fill_away(away_col)
            upcoming[col] = (
                pd.to_numeric(upcoming[home_col], errors="coerce").fillna(medians.get(home_col, 0.0))
                - pd.to_numeric(upcoming[away_col], errors="coerce").fillna(medians.get(away_col, 0.0))
            )
        else:
            upcoming[col] = medians.get(col, 0.0)
    return upcoming
