"""market.py — de-vigged bookmaker probabilities and the market-anchored blend.

Bookmaker closing prices are the strongest single predictor of football
outcomes. The base classifier deliberately stays market-blind; this module
combines its probabilities with the market (and the Dixon-Coles model) through
a learned out-of-fold stacker, so the served prediction is anchored to the
market without leaking the same-match result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Model class order is 0=Away, 1=Draw, 2=Home — market columns follow it.
MARKET_COLS = ["market_prob_away", "market_prob_draw", "market_prob_home"]
_ODDS_BY_CLASS = ("B365A", "B365D", "B365H")
ODDS_COLS = ("B365H", "B365D", "B365A")


def market_probs_from_odds(df: pd.DataFrame) -> pd.DataFrame | None:
    """Return proportional de-vigged probabilities from B365H/D/A.

    Columns are ordered like the model classes (away, draw, home) so the matrix
    can be passed straight into the blend. Rows with missing/invalid odds are
    NaN so callers can fall back to model-only probabilities per fixture.
    Returns ``None`` when the odds columns are absent entirely.
    """
    if not all(col in df.columns for col in ODDS_COLS):
        return None
    odds = df[list(_ODDS_BY_CLASS)].apply(pd.to_numeric, errors="coerce")
    valid = (odds > 1.0).all(axis=1)
    inv = 1.0 / odds
    total = inv.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = inv.div(total, axis=0)
    probs[~valid] = np.nan
    probs.columns = MARKET_COLS
    return probs


def blend_matrix(model_probs, dc_probs, market_probs) -> np.ndarray:
    """Stack ``[model, dixon-coles, market]`` probability triplets as 9 columns."""
    model_probs = np.asarray(model_probs, dtype=float)
    dc_probs = np.asarray(dc_probs, dtype=float)
    market_probs = np.asarray(market_probs, dtype=float)
    if model_probs.shape != dc_probs.shape or model_probs.shape != market_probs.shape:
        raise ValueError(
            f"blend inputs must share shape: model={model_probs.shape}, "
            f"dc={dc_probs.shape}, market={market_probs.shape}"
        )
    return np.hstack([model_probs, dc_probs, market_probs])


def fit_market_blend(model_probs, dc_probs, market_probs, y, *, l2: float = 1e-3):
    """Fit the out-of-fold market stacker + per-class (vector) calibration."""
    from model_artifacts import MarketBlendModel

    X = blend_matrix(model_probs, dc_probs, market_probs)
    return MarketBlendModel().fit(X, np.asarray(y, dtype=int), l2=l2)


def evaluate_probs(probs: np.ndarray, y: np.ndarray) -> dict:
    """Small shared metric bundle used by the backtest and the retrain gate."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(y)
    if n == 0:
        return {"n": 0, "log_loss": float("nan"), "accuracy": float("nan")}
    p_true = np.clip(probs[np.arange(n), y], 1e-12, None)
    return {
        "n": int(n),
        "log_loss": float(-np.mean(np.log(p_true))),
        "accuracy": float(np.mean(np.argmax(probs, axis=1) == y)),
    }
