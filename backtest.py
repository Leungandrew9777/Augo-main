#!/usr/bin/env python3
"""backtest.py — season rolling-origin backtest with market-relative baselines.

For each of the last N seasons the model is trained only on earlier seasons
(rolling origin), optionally with nested hyperparameter tuning, and evaluated
on the held-out season. Reports the base classifier, Dixon-Coles, market-only,
base-rate and the market blend on the same rows, with bootstrap confidence
intervals for the blend-vs-market log-loss difference and per-class reliability.

Usage:
    python backtest.py                    # last 5 full seasons + current, 1 seed
    python backtest.py --seasons 5 --seeds 42,7,2026
    python backtest.py --tune --trials 4
    python backtest.py --json             # also write backtest_report.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

from features import MODEL_FEATURE_COLS
from market import MARKET_COLS, fit_market_blend, market_probs_from_odds
from metrics import (
    brier_multiclass,
    expected_calibration_error,
    per_class_ece,
)
from model_artifacts import StackedEnsembleClassifier, TemperatureScalingCalibrator

APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_FILE = os.path.join(APP_DIR, "backtest_report.json")


def season_start_year(season: Any) -> int:
    text = str(season)
    return int(text[:2]) + 2000


def season_test_folds(df: pd.DataFrame, *, n_seasons: int = 5) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Return ``(season, train, test)`` rolling-origin folds."""
    if "Season" not in df.columns:
        raise ValueError("backtest needs a Season column (run data_pipeline + feature_engineering)")
    work = df.copy()
    work["_season"] = work["Season"].astype(str)
    counts = work["_season"].value_counts()
    ordered = sorted(counts.index.tolist(), key=season_start_year)
    full = [season for season in ordered if counts[season] >= 200]
    test_seasons = full[-n_seasons:] if n_seasons > 0 else full
    current = ordered[-1] if ordered else None
    if current and current not in test_seasons and counts.get(current, 0) >= 30:
        test_seasons = test_seasons + [current]

    folds: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for season in test_seasons:
        start = season_start_year(season)
        train = work[work["_season"].map(season_start_year) < start].drop(columns=["_season"])
        test = work[work["_season"] == season].drop(columns=["_season"])
        if train.empty or test.empty:
            continue
        folds.append((season, train, test))
    return folds


def _class_weights(y: np.ndarray) -> np.ndarray:
    counts = pd.Series(y).value_counts().sort_index()
    weights = len(y) / (3.0 * counts.reindex([0, 1, 2]).astype(float).to_numpy())
    return weights


def _metrics(probs: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(y)
    p_true = np.clip(probs[np.arange(n), y], 1e-12, None)
    ece, _rows = expected_calibration_error(probs, y)
    return {
        "n": int(n),
        "log_loss": float(log_loss(y, probs, labels=[0, 1, 2])),
        "brier": float(brier_multiclass(probs, y)),
        "accuracy": float(np.mean(np.argmax(probs, axis=1) == y)),
        "ece": float(ece),
        "per_class_ece": {str(k): float(v) for k, v in per_class_ece(probs, y).items()},
    }


def _bootstrap_mean_ci(values: np.ndarray, *, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _tune_params(train_df, X_tr, y_tr, class_weight, trials: int):
    from train_ensemble import _sample_params, walkforward_metrics

    rng = np.random.default_rng(42)
    best: tuple[float, dict] | None = None
    for _ in range(max(1, trials)):
        params = _sample_params(rng)
        loss, _acc = walkforward_metrics(
            X_tr, pd.Series(y_tr), train_df["Date"].reset_index(drop=True),
            params, class_weight, n_splits=3,
        )
        if best is None or loss < best[0]:
            best = (loss, params)
    print(f"   nested tuning selected logloss={best[0]:.4f} params={best[1]}")
    return best[1]


def _fit_predict_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
    *,
    seeds: tuple[int, ...],
    tune_trials: int = 0,
) -> dict[str, Any]:
    from train_ensemble import (
        base_meta_input,
        fit_base,
        fit_meta,
        soft_vote_probs,
        time_decay_weights,
    )
    from dixon_coles import DixonColesModel

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    medians = train_df[feature_cols].median()
    X_tr = train_df[feature_cols].fillna(medians)
    X_te = test_df[feature_cols].fillna(medians)
    y_tr = train_df["Result"].astype(int).to_numpy()
    y_te = test_df["Result"].astype(int).to_numpy()
    class_weight = _class_weights(y_tr)

    if tune_trials:
        params = _tune_params(train_df, X_tr, pd.Series(y_tr), class_weight, tune_trials)

    # Nested inner OOF for the meta-learner, blend and calibration.
    inner = TimeSeriesSplit(n_splits=3)
    n_tr = len(X_tr)
    meta_X = np.full((n_tr, 9), np.nan)
    vote_inner = np.full((n_tr, 3), np.nan)
    dc_inner = np.full((n_tr, 3), np.nan)
    for itr, ite in inner.split(X_tr):
        dates_in = train_df["Date"].iloc[itr]
        decay_in = time_decay_weights(dates_in)
        sw_in = decay_in * class_weight[y_tr[itr]]
        base_in = fit_base(X_tr.iloc[itr], pd.Series(y_tr[itr]), sw_in, params, seeds=seeds)
        meta_X[ite] = base_meta_input(base_in, X_tr.iloc[ite])
        vote_inner[ite] = soft_vote_probs(base_in, X_tr.iloc[ite])
        dc_fold = DixonColesModel().fit(train_df.iloc[itr])
        dc_inner[ite] = dc_fold.predict_proba(train_df.iloc[ite])

    filled = ~np.isnan(meta_X).any(axis=1)
    meta = fit_meta(meta_X[filled], pd.Series(y_tr[filled]))
    stacked_inner = np.full((n_tr, 3), np.nan)
    stacked_inner[filled] = (
        0.5 * meta.predict_proba(meta_X[filled]) + 0.5 * vote_inner[filled]
    )
    calibrator = TemperatureScalingCalibrator().fit(stacked_inner[filled], y_tr[filled])
    model_inner = calibrator.transform(stacked_inner[filled])

    market_train = market_probs_from_odds(train_df)
    if market_train is None:
        market_train = pd.DataFrame(np.nan, index=train_df.index, columns=MARKET_COLS)
    market_tr_np = market_train.to_numpy()
    inner_ok = filled & np.isfinite(market_tr_np).all(axis=1)
    blend = None
    if inner_ok.sum() >= 100:
        blend = fit_market_blend(
            model_inner[inner_ok[filled]],
            dc_inner[filled][inner_ok[filled]],
            market_tr_np[inner_ok],
            y_tr[filled][inner_ok[filled]],
        )

    # Final fit on the whole training window, then predict the held-out season.
    decay = time_decay_weights(train_df["Date"])
    sw = decay * class_weight[y_tr]
    base = fit_base(X_tr, pd.Series(y_tr), sw, params, seeds=seeds)
    meta_final = fit_meta(meta_X[filled], pd.Series(y_tr[filled]))
    stacked = StackedEnsembleClassifier(
        estimators_=base,
        meta_estimator_=meta_final,
        classes_=np.array([0, 1, 2]),
        feature_names_in_=np.array(list(X_tr.columns), dtype=object),
        vote_weight=0.5,
    )
    stacked.calibrator_ = calibrator

    model_te = np.asarray(stacked.predict_proba(X_te), dtype=float)
    dc_te = DixonColesModel().fit(train_df).predict_proba(test_df)

    market_test = market_probs_from_odds(test_df)
    if market_test is None:
        market_test = pd.DataFrame(np.nan, index=test_df.index, columns=MARKET_COLS)
    market_te = market_test.to_numpy()
    ok = np.isfinite(market_te).all(axis=1)

    blend_te = model_te.copy()
    if blend is not None and ok.any():
        dc_part = dc_te[ok]
        blend_te[ok] = blend.predict_proba(model_te[ok], dc_part, market_te[ok])

    base_rate = np.tile((np.bincount(y_tr, minlength=3) / len(y_tr)), (len(y_te), 1))
    return {
        "y": y_te,
        "model": model_te,
        "dc": dc_te,
        "market": market_te,
        "blend": blend_te,
        "base_rate": base_rate,
        "market_ok": ok,
    }


def _saved_params() -> dict:
    from train_ensemble import BEST_PARAMS_FILE, DEFAULT_PARAMS

    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_PARAMS


def run_backtest(
    df: pd.DataFrame | None = None,
    *,
    n_seasons: int = 5,
    seeds: tuple[int, ...] = (42,),
    tune_trials: int = 0,
) -> dict[str, Any]:
    from train_ensemble import load_training_frame

    df = df if df is not None else load_training_frame()
    feature_cols = [c for c in MODEL_FEATURE_COLS if c in df.columns]
    params = _saved_params()
    folds = season_test_folds(df, n_seasons=n_seasons)
    if not folds:
        raise SystemExit("backtest: no season folds available")

    results: list[dict[str, Any]] = []
    for season, train_df, test_df in folds:
        print(f"\n=== test season {season}: train={len(train_df):,} test={len(test_df):,} ===")
        preds = _fit_predict_fold(
            train_df, test_df, feature_cols, params,
            seeds=seeds, tune_trials=tune_trials,
        )
        ok = preds["market_ok"]
        y = preds["y"][ok]
        rows = {
            "season": season,
            "n_all": int(len(preds["y"])),
            "n_market": int(ok.sum()),
            "base": _metrics(preds["model"][ok], y),
            "dc": _metrics(preds["dc"][ok], y),
            "market": _metrics(preds["market"][ok], y),
            "blend": _metrics(preds["blend"][ok], y),
            "base_rate": _metrics(preds["base_rate"][ok], y),
        }
        # Per-match log-loss difference: blend minus market (negative = blend better).
        y_idx = y
        eps = 1e-12
        blend_ll = -np.log(np.clip(preds["blend"][ok][np.arange(len(y)), y_idx], eps, None))
        market_ll = -np.log(np.clip(preds["market"][ok][np.arange(len(y)), y_idx], eps, None))
        diff = blend_ll - market_ll
        lo, hi = _bootstrap_mean_ci(diff)
        rows["blend_minus_market_ci"] = [lo, hi]
        rows["blend_minus_market_values"] = diff.tolist()
        results.append(rows)
        print(f"   n={rows['n_market']} | base={rows['base']['log_loss']:.4f} "
              f"dc={rows['dc']['log_loss']:.4f} market={rows['market']['log_loss']:.4f} "
              f"blend={rows['blend']['log_loss']:.4f} | "
              f"blend-market 95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"   blend accuracy={rows['blend']['accuracy']:.3f} | "
              f"ECE={rows['blend']['ece']:.3f} | per-class ECE={rows['blend']['per_class_ece']}")

    pooled = _pooled_metrics(results)
    print("\n=== pooled (all market rows) ===")
    for key in ("base_rate", "market", "base", "dc", "blend"):
        m = pooled[key]
        print(f"   {key:<10} n={m['n']} logloss={m['log_loss']:.4f} "
              f"brier={m['brier']:.4f} acc={m['accuracy']:.3f} ece={m['ece']:.3f}")
    print(f"   blend-minus-market mean={pooled['blend_minus_market_mean']:+.4f} "
          f"95% CI [{pooled['blend_minus_market_ci'][0]:+.4f}, "
          f"{pooled['blend_minus_market_ci'][1]:+.4f}]")

    return {"folds": results, "pooled": pooled}


def _pooled_metrics(folds: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate by pooling per-match losses across seasons (not mean of means)."""
    rows = [f for f in folds if f["n_market"] > 0]
    if not rows:
        return {"n": 0}
    total = sum(r["n_market"] for r in rows)
    pooled: dict[str, Any] = {}
    for key in ("base_rate", "market", "base", "dc", "blend"):
        pooled[key] = {
            "n": total,
            "log_loss": float(sum(r[key]["log_loss"] * r["n_market"] for r in rows) / total),
            "brier": float(sum(r[key]["brier"] * r["n_market"] for r in rows) / total),
            "accuracy": float(sum(r[key]["accuracy"] * r["n_market"] for r in rows) / total),
            "ece": float(sum(r[key]["ece"] * r["n_market"] for r in rows) / total),
        }
    values = np.asarray(
        [value for row in rows for value in row.get("blend_minus_market_values", [])],
        dtype=float,
    )
    pooled["blend_minus_market_mean"] = float(np.mean(values)) if len(values) else float("nan")
    pooled["blend_minus_market_ci"] = list(_bootstrap_mean_ci(values))
    return pooled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Season rolling-origin backtest")
    parser.add_argument("--seasons", type=int, default=5, help="number of full seasons to test")
    parser.add_argument("--seeds", default="42", help="comma-separated seeds, e.g. 42,7,2026")
    parser.add_argument("--tune", action="store_true", help="nested hyperparameter search per fold")
    parser.add_argument("--trials", type=int, default=4, help="nested tuning trials per fold")
    parser.add_argument("--json", action="store_true", help="write backtest_report.json")
    args = parser.parse_args(argv)

    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    report = run_backtest(n_seasons=args.seasons, seeds=seeds, tune_trials=args.trials if args.tune else 0)

    if args.json:
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written -> {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
