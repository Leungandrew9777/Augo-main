import json
import os

import numpy as np
import pandas as pd
import pytest

import dates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "xgboost_premier_league_model.pkl")
ELO = os.path.join(ROOT, "premier_league_with_elo_best.csv")
STATE = os.path.join(ROOT, "team_state.json")
FIXTURES = os.path.join(ROOT, "fixtures.csv")
RESULTS = os.path.join(ROOT, "results.csv")

pytestmark = pytest.mark.skipif(
    not all(os.path.exists(path) for path in (MODEL, ELO, STATE, FIXTURES)),
    reason="model/data artifacts not built",
)


def _next_gameweek(fixtures: pd.DataFrame) -> int:
    future = fixtures[fixtures["_date"] >= pd.Timestamp(dates.today_uk())].sort_values("_date")
    if not future.empty:
        return int(future.iloc[0]["gameweek"])
    return int(pd.to_numeric(fixtures["gameweek"], errors="coerce").dropna().max())


def test_real_artifacts_predict_next_gameweek():
    from features import (
        MODEL_FEATURE_COLS,
        compute_current_elo,
        detect_model_features,
        ensure_model_features,
        load_model_and_elo,
    )
    from inference import predict_matches

    model, df_elo = load_model_and_elo(MODEL, ELO)
    fixtures = pd.read_csv(FIXTURES)
    fixtures["_date"] = dates.parse_series(fixtures["date"])
    fixtures = fixtures.dropna(subset=["_date", "home_team", "away_team"])

    gw = _next_gameweek(fixtures)
    selected = fixtures[pd.to_numeric(fixtures["gameweek"], errors="coerce") == gw].sort_values("_date")
    upcoming = pd.DataFrame({
        "date": selected["_date"].dt.strftime("%Y-%m-%d"),
        "home_team": selected["home_team"].astype(str),
        "away_team": selected["away_team"].astype(str),
    }).reset_index(drop=True)

    upcoming = compute_current_elo(upcoming, df_elo)
    feature_cols = detect_model_features(model) or MODEL_FEATURE_COLS
    upcoming = ensure_model_features(upcoming, df_elo, feature_cols)
    frame = upcoming[feature_cols]

    assert list(frame.columns) == feature_cols
    assert all(pd.api.types.is_numeric_dtype(frame[col]) for col in feature_cols)
    assert not frame.isna().any().any()

    out = predict_matches(upcoming, model, df_elo)
    probs = out[["prob_home", "prob_draw", "prob_away"]].to_numpy(dtype=float)
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_team_state_is_not_stale():
    state = json.load(open(STATE, encoding="utf-8"))
    assert state["last_match_date"]
    if not os.path.exists(RESULTS):
        return
    results = pd.read_csv(RESULTS)
    played = dates.parse_series(results["date"]).dropna()
    if played.empty:
        return
    assert state["last_match_date"] >= str(played.max().date())
