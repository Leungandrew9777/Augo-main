import numpy as np
import pandas as pd
import pytest

from market import blend_matrix, fit_market_blend, market_probs_from_odds
from model_artifacts import MarketBlendModel, VectorScalingCalibrator


def test_devig_probs_order_and_sum():
    df = pd.DataFrame({"B365H": [2.0], "B365D": [4.0], "B365A": [4.0]})
    probs = market_probs_from_odds(df)
    assert probs is not None
    assert probs.loc[0, "market_prob_home"] == pytest.approx(0.5)
    assert probs.loc[0, "market_prob_draw"] == pytest.approx(0.25)
    assert probs.loc[0, "market_prob_away"] == pytest.approx(0.25)
    assert probs.loc[0].sum() == pytest.approx(1.0)


def test_devig_missing_odds_nan_and_missing_columns():
    df = pd.DataFrame({"B365H": [2.0, None], "B365D": [4.0, 4.0], "B365A": [4.0, 4.0]})
    probs = market_probs_from_odds(df)
    assert np.isnan(probs.loc[1]).all()
    assert market_probs_from_odds(pd.DataFrame({"B365H": [2.0]})) is None


def test_blend_matrix_shape():
    model = np.array([[0.5, 0.3, 0.2]])
    dc = np.array([[0.4, 0.3, 0.3]])
    market = np.array([[0.45, 0.3, 0.25]])
    X = blend_matrix(model, dc, market)
    assert X.shape == (1, 9)


def test_market_blend_outputs_probabilities():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 3, size=n)
    base = rng.dirichlet([2, 2, 2], size=n)
    dc = rng.dirichlet([2, 2, 2], size=n)
    raw = base + dc + rng.normal(0, 0.02, size=(n, 3))
    market = np.clip(raw / raw.sum(axis=1, keepdims=True), 1e-6, None)
    model = fit_market_blend(base, dc, market, y)
    assert isinstance(model, MarketBlendModel)
    probs = model.predict_proba(base, dc, market)
    assert probs.shape == (n, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    cal = VectorScalingCalibrator().fit(probs, y)
    out = cal.transform(probs)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)
    assert cal.scale_.shape == (3,)
