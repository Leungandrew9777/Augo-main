import numpy as np
import pandas as pd
import pytest

from dixon_coles import DixonColesModel, hda_from_lambdas


def _history() -> pd.DataFrame:
    matches = [
        ("2026-08-01", "A", "B", 2, 0),
        ("2026-08-08", "B", "A", 1, 1),
        ("2026-08-15", "A", "B", 3, 1),
        ("2026-08-22", "B", "A", 0, 2),
        ("2026-08-29", "A", "B", 1, 0),
        ("2026-09-05", "B", "A", 2, 2),
        ("2026-09-12", "A", "B", 0, 1),
        ("2026-09-19", "B", "A", 1, 0),
    ]
    return pd.DataFrame([
        {
            "Date": pd.Timestamp(day), "HomeTeam": home, "AwayTeam": away,
            "FTHG": home_goals, "FTAG": away_goals,
            "home_xg": np.nan, "away_xg": np.nan,
        }
        for day, home, away, home_goals, away_goals in matches
    ])


def test_hda_from_lambdas_ordering_and_sum():
    probs = hda_from_lambdas(np.array([3.0, 0.2]), np.array([0.2, 3.0]), rho=-0.05)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    # Class order is [Away, Draw, Home].
    assert probs[0, 2] > probs[0, 0]
    assert probs[1, 0] > probs[1, 2]


def test_fit_predict_shape_and_bounds():
    model = DixonColesModel().fit(_history())
    probs = model.predict_proba(_history())
    assert probs.shape == (len(_history()), 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert 0.0 <= model.home_adv_ <= 1.5
    assert set(model.teams_) == {"A", "B"}


def test_unknown_team_uses_league_average():
    model = DixonColesModel().fit(_history())
    fixture = pd.DataFrame([{"HomeTeam": "Promoted", "AwayTeam": "A"}])
    probs = model.predict_proba(fixture)
    assert probs.shape == (1, 3)
    assert np.isclose(probs.sum(), 1.0)
