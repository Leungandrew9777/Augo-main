"""
features.py — single source of truth for match-feature construction.

Training (train_ensemble.py) and every inference path (run_pipeline.py,
app.py) build features through this module so the feature set can never
drift apart between the fitted model and live predictions.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from team_aliases import elo_lookup_key

# Serving snapshot written by feature_engineering.py (post-last-match ELO +
# per-team/venue rolling features). Falls back to the historical CSV if absent.
TEAM_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "team_state.json")
_TEAM_STATE_CACHE: dict[str, tuple[float | None, dict | None]] = {}


def load_team_state(path: str | None = None) -> dict | None:
    """Load the team-state snapshot, cached until its mtime changes."""
    target = path or TEAM_STATE_FILE
    try:
        mtime = os.path.getmtime(target)
    except OSError:
        return None
    cached = _TEAM_STATE_CACHE.get(target)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    _TEAM_STATE_CACHE[target] = (mtime, data)
    return data

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

    def _patch(est) -> None:
        if hasattr(est, "named_steps") and "model" in est.named_steps:
            est = est.named_steps["model"]
        if isinstance(est, LogisticRegression) and not hasattr(est, "multi_class"):
            setattr(est, "multi_class", "auto")

    for est in getattr(model, "estimators_", []):
        _patch(est)
    # Stacked ensembles keep the meta learner outside ``estimators_``; patch it too.
    meta = getattr(model, "meta_estimator_", None)
    if meta is not None:
        _patch(meta)


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


def _sorted_elo_history(df_elo: pd.DataFrame) -> pd.DataFrame:
    """Date-sorted copy of the ELO history (empty if it has no usable date)."""
    if "date" not in df_elo.columns:
        return df_elo.iloc[0:0]
    work = df_elo.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    return work.dropna(subset=["date"]).sort_values("date", kind="stable")


def latest_venue_rows(
    df_elo: pd.DataFrame, venue_col: str, feature_cols: list[str],
) -> dict[str, dict[str, float]]:
    """Latest non-NaN value of each feature per team at the given venue.

    A single groupby pass replaces the per-column/per-fixture frame filtering
    the inference paths used to do. ``GroupBy.last`` keeps the original
    "most recent non-NaN value" semantics.
    """
    if venue_col not in df_elo.columns:
        return {}
    cols = [c for c in feature_cols if c in df_elo.columns]
    if not cols:
        return {}
    work = _sorted_elo_history(df_elo)
    if work.empty or "date" not in work.columns:
        return {}
    return work.groupby(venue_col, sort=False)[cols].last().to_dict("index")


HOME_ADVANTAGE_FALLBACK = 65.0


def compute_current_elo(
    upcoming: pd.DataFrame,
    df_elo: pd.DataFrame,
    elo_key=elo_lookup_key,
) -> pd.DataFrame:
    """Attach the current ELO of each side and the ELO difference.

    The serving snapshot (``team_state.json``) holds ratings *after* the last
    completed match, so predictions include that result instead of lagging one
    match behind. The home side gets the home-advantage bump exactly once,
    matching the ``elo_diff`` the models were trained on.
    """
    state = load_team_state()
    ratings = state.get("ratings") if isinstance(state, dict) else None
    if isinstance(ratings, dict) and ratings:
        ha = float(state.get("home_advantage", HOME_ADVANTAGE_FALLBACK))

        def _rating(team) -> float:
            key = elo_key(str(team))
            return float(ratings.get(key, ratings.get(str(team), 1500.0)))

        upcoming["elo_home"] = upcoming["home_team"].map(lambda t: _rating(t) + ha)
        upcoming["elo_away"] = upcoming["away_team"].map(_rating)
        upcoming["elo_diff"] = upcoming["elo_home"] - upcoming["elo_away"]
        return upcoming

    work = _sorted_elo_history(df_elo)
    if {"elo_home_rating", "elo_away_rating"}.issubset(work.columns):
        home_col, away_col = "elo_home_rating", "elo_away_rating"
        advantage = work["elo_home_before"] - work["elo_home_rating"]
        ha = float(advantage.median()) if advantage.notna().any() else HOME_ADVANTAGE_FALLBACK
    else:
        home_col, away_col = "elo_home_before", "elo_away_before"
        ha = 0.0

    home = work[["date", "home_team", home_col]].rename(
        columns={"home_team": "team", home_col: "elo"}
    )
    away = work[["date", "away_team", away_col]].rename(
        columns={"away_team": "team", away_col: "elo"}
    )
    long = pd.concat([home, away], ignore_index=True).dropna(subset=["elo"])
    long = long.sort_values("date", kind="stable")
    latest_rating: dict[str, float] = long.groupby("team", sort=False)["elo"].last().to_dict()

    upcoming["elo_home"] = upcoming["home_team"].map(
        lambda t: latest_rating.get(elo_key(str(t)), 1500.0) + ha
    )
    upcoming["elo_away"] = upcoming["away_team"].map(
        lambda t: latest_rating.get(elo_key(str(t)), 1500.0)
    )
    upcoming["elo_diff"] = upcoming["elo_home"] - upcoming["elo_away"]
    return upcoming


def _feature_names(obj) -> list[str]:
    names = getattr(obj, "feature_names_in_", None)
    if names is not None and len(names):
        return [str(c) for c in list(names)]
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            found = _feature_names(step)
            if found:
                return found
    return []


def detect_model_features(model) -> list[str]:
    """Extract expected feature names from a fitted model/ensemble (or [])."""
    names = _feature_names(model)
    if names:
        return names
    for est in list(getattr(model, "estimators_", []))[:1]:
        names = _feature_names(est)
        if names:
            return names
    return []


def ensure_model_features(
    upcoming: pd.DataFrame,
    df_elo: pd.DataFrame,
    expected_cols: list[str],
) -> pd.DataFrame:
    """Fill every column the model expects, per fixture.

    Home/away features come from the post-last-match snapshot
    (``team_state.json``) so the most recent result is included; if the
    snapshot is missing the newest matching venue row in the ELO history is
    used instead. ``diff_*`` are derived and H2H features are computed per
    fixture from prior meetings (not a global median).
    """
    if not expected_cols:
        return upcoming

    missing = [c for c in expected_cols if c not in upcoming.columns]
    if not missing and all(
        pd.api.types.is_numeric_dtype(upcoming[c])
        and not pd.to_numeric(upcoming[c], errors="coerce").isna().any()
        for c in expected_cols
    ):
        # Already fully populated (later model families reuse the same columns);
        # skip the medians/H2H recomputation.
        return upcoming

    medians: dict[str, float] = {}
    for col in expected_cols:
        if col in df_elo.columns:
            s = pd.to_numeric(df_elo[col], errors="coerce").dropna()
            if not s.empty:
                medians[col] = float(s.median())

    home_latest = latest_venue_rows(df_elo, "home_team", expected_cols)
    away_latest = latest_venue_rows(df_elo, "away_team", expected_cols)

    state = load_team_state()
    state_teams = state.get("teams") if isinstance(state, dict) else None
    if not isinstance(state_teams, dict):
        state_teams = {}

    def _latest(rows: dict[str, dict[str, float]], team, col: str):
        row = rows.get(elo_lookup_key(str(team)))
        if row is None:
            return None
        return row.get(col)

    def _state_value(team, venue: str, suffix: str):
        key = elo_lookup_key(str(team))
        entry = state_teams.get(key) or state_teams.get(str(team))
        if not entry:
            return None
        return (entry.get(venue) or {}).get(suffix)

    new_cols: dict[str, pd.Series] = {}

    def _series_for(col: str) -> pd.Series:
        if col in new_cols:
            return new_cols[col]
        return pd.to_numeric(upcoming[col], errors="coerce")

    def _fill_home(col: str):
        fallback = medians.get(col, 0.0)
        if state_teams:
            suffix = col[len("home_"):]
            mapped = upcoming["home_team"].map(lambda t: _state_value(t, "home", suffix))
        else:
            mapped = upcoming["home_team"].map(lambda t: _latest(home_latest, t, col))
        new_cols[col] = pd.to_numeric(mapped, errors="coerce").fillna(fallback)

    def _fill_away(col: str):
        fallback = medians.get(col, 0.0)
        if state_teams:
            suffix = col[len("away_"):]
            mapped = upcoming["away_team"].map(lambda t: _state_value(t, "away", suffix))
        else:
            mapped = upcoming["away_team"].map(lambda t: _latest(away_latest, t, col))
        new_cols[col] = pd.to_numeric(mapped, errors="coerce").fillna(fallback)

    if "date" in upcoming.columns and any(c in expected_cols for c in (
        "h2h_home_wins", "h2h_draws", "h2h_total_goals_avg",
    )):
        def _h2h_features(row) -> pd.Series:
            home = elo_lookup_key(str(row["home_team"]))
            away = elo_lookup_key(str(row["away_team"]))
            date = pd.to_datetime(row["date"], errors="coerce")
            if pd.isna(date):
                return pd.Series({
                    "h2h_home_wins": np.nan, "h2h_draws": np.nan, "h2h_total_goals_avg": np.nan,
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
                    "h2h_home_wins": np.nan, "h2h_draws": np.nan, "h2h_total_goals_avg": np.nan,
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
            # A later model family (e.g. corner models) can run this again after
            # H2H was already filled; never add a duplicate column name.
            if col not in upcoming.columns and col not in new_cols:
                new_cols[col] = pd.to_numeric(h2h[col], errors="coerce")

    # All fills are collected first and appended in one concat: assigning ~40
    # columns one by one fragments the frame and is much slower.
    for col in expected_cols:
        if col in new_cols:
            new_cols[col] = pd.to_numeric(new_cols[col], errors="coerce").fillna(
                medians.get(col, 0.0)
            )
        elif col in upcoming.columns:
            upcoming[col] = pd.to_numeric(upcoming[col], errors="coerce").fillna(medians.get(col, 0.0))
        elif col.startswith("home_"):
            _fill_home(col)
        elif col.startswith("away_"):
            _fill_away(col)
        elif col.startswith("diff_"):
            suffix = col[len("diff_"):]
            home_col = f"home_{suffix}"
            away_col = f"away_{suffix}"
            if home_col not in upcoming.columns and home_col not in new_cols:
                _fill_home(home_col)
            if away_col not in upcoming.columns and away_col not in new_cols:
                _fill_away(away_col)
            new_cols[col] = (
                _series_for(home_col).fillna(medians.get(home_col, 0.0))
                - _series_for(away_col).fillna(medians.get(away_col, 0.0))
            )
        else:
            new_cols[col] = pd.Series(medians.get(col, 0.0), index=upcoming.index)

    if new_cols:
        upcoming = pd.concat([upcoming, pd.DataFrame(new_cols, index=upcoming.index)], axis=1)

    # XGBoost's DMatrix rejects object columns and pandas 3 no longer
    # downcasts object -> float on fillna, so pin every model feature to a
    # float column to keep serving identical on pandas 2 and pandas 3.
    for col in expected_cols:
        if col in upcoming.columns:
            upcoming[col] = pd.to_numeric(upcoming[col], errors="coerce").astype(float)
    return upcoming
