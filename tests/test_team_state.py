import json

import pandas as pd
import pytest

import features
from feature_engineering import FootballELO, build_team_state
from features import compute_current_elo, ensure_model_features, load_team_state


def _history() -> pd.DataFrame:
    matches = [
        ("2026-08-01", "A", "B", 2, 0),
        ("2026-08-08", "B", "A", 1, 1),
        ("2026-08-15", "A", "B", 3, 1),
        ("2026-08-22", "B", "A", 0, 2),
        ("2026-08-29", "A", "B", 1, 0),
        ("2026-09-05", "B", "A", 2, 2),
    ]
    return pd.DataFrame([
        {
            "Date": pd.Timestamp(day),
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": home_goals,
            "FTAG": away_goals,
            "HS": 10, "AS": 8, "HST": 5, "AST": 4,
            "HC": 6, "AC": 4, "HF": 10, "AF": 12,
            "home_xg": 1.5, "away_xg": 1.0,
            "elo_home_rating": 1500.0, "elo_away_rating": 1500.0,
        }
        for day, home, away, home_goals, away_goals in matches
    ])


def _final_elo() -> FootballELO:
    elo = FootballELO(k=32, home_advantage=65)
    elo.ratings = {"A": 1600.0, "B": 1400.0}
    return elo


def _state() -> dict:
    return build_team_state(_history(), _final_elo(), window=5)


def test_state_includes_last_match_and_final_elo():
    state = _state()
    assert state["last_match_date"] == "2026-09-05"
    assert state["ratings"] == {"A": 1600.0, "B": 1400.0}

    # A's home games: 2-0 W, 3-1 W, 1-0 W (opponent always 1500 -> factor 1.0).
    assert state["teams"]["A"]["home"]["avg_GF"] == pytest.approx(2.0)
    assert state["teams"]["A"]["home"]["Form"] == pytest.approx(3.0)
    # B's away games: 0, 1, 0 goals; 0 points.
    assert state["teams"]["B"]["away"]["avg_GF"] == pytest.approx(1.0 / 3.0)
    assert state["teams"]["B"]["away"]["Form"] == pytest.approx(0.0)


def test_state_round_trip(tmp_path):
    path = tmp_path / "team_state.json"
    path.write_text(json.dumps(_state()), encoding="utf-8")
    loaded = load_team_state(str(path))
    assert loaded is not None
    assert loaded["ratings"]["A"] == pytest.approx(1600.0)


def test_compute_current_elo_uses_state(monkeypatch):
    monkeypatch.setattr(features, "load_team_state", lambda path=None: _state())
    upcoming = pd.DataFrame([{"home_team": "A", "away_team": "B"}])
    out = compute_current_elo(upcoming, pd.DataFrame())
    assert out.loc[0, "elo_home"] == pytest.approx(1600.0 + 65.0)
    assert out.loc[0, "elo_away"] == pytest.approx(1400.0)
    assert out.loc[0, "elo_diff"] == pytest.approx(265.0)


def test_ensure_model_features_uses_state(monkeypatch):
    monkeypatch.setattr(features, "load_team_state", lambda path=None: _state())
    upcoming = pd.DataFrame([{"home_team": "A", "away_team": "B"}])
    expected = ["home_Form", "away_Form", "home_avg_GF", "away_avg_GF", "diff_avg_GF"]
    out = ensure_model_features(upcoming, _history(), expected)
    assert out.loc[0, "home_avg_GF"] == pytest.approx(2.0)
    assert out.loc[0, "away_avg_GF"] == pytest.approx(1.0 / 3.0)
    assert out.loc[0, "diff_avg_GF"] == pytest.approx(2.0 - 1.0 / 3.0)
    assert out.loc[0, "home_Form"] == pytest.approx(3.0)
