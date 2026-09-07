#!/usr/bin/env python3
"""
evaluate.py — probabilistic + ROI evaluation of archived predictions vs results.

Reads every predictions_history/GW{N}.json, joins it with results.csv (via
persistence.py) and reports, per gameweek and cumulative:
  - accuracy, log-loss, Brier score, expected calibration error (ECE)
  - ROI betting the model pick at (a) model fair odds and (b) bookmaker odds
  - ROI of a "value" strategy (model pick at book odds only when the model
    probability beats the bookmaker implied probability)
  - a calibration table (confidence bins vs empirical hit rate)

Usage:
    python evaluate.py          # print report
    python evaluate.py --json   # also write evaluation_report.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd

from persistence import load_archived_predictions, load_results, load_user_picks
from team_aliases import fixture_lookup_key

APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_FILE = os.path.join(APP_DIR, "evaluation_report.json")

OUTCOMES = ("H", "D", "A")
OUTCOME_KEY = {"H": "home", "D": "draw", "A": "away"}


def _f(v: Any, default: float | None = None) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def brier_multiclass(probs: np.ndarray, y_true: np.ndarray) -> float:
    n = len(y_true)
    one_hot = np.zeros((n, probs.shape[1]))
    one_hot[np.arange(n), y_true] = 1.0
    return float(np.mean((probs - one_hot) ** 2))


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, bins: int = 10) -> float:
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == y_true).astype(float)
    ece = 0.0
    rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        if mask.sum() == 0:
            continue
        acc = float(correct[mask].mean())
        avg = float(conf[mask].mean())
        ece += (mask.sum() / len(y_true)) * abs(acc - avg)
        rows.append((mask.sum(), acc, avg))
    return float(ece), rows


def _money(v: float) -> str:
    return f"{v:+.2f}u" if v >= 0 else f"{v:.2f}u"


def _collect() -> tuple[list[dict[str, Any]], dict[int, dict[int, str]]]:
    archives = load_archived_predictions()
    results = load_results()
    picks = load_user_picks()

    records: list[dict[str, Any]] = []
    for gw in sorted(archives.keys()):
        preds = archives[gw].get("predictions", []) if isinstance(archives[gw], dict) else []
        for i, p in enumerate(preds):
            if not isinstance(p, dict):
                continue
            home = str(p.get("home_team", ""))
            away = str(p.get("away_team", ""))
            key = (gw, fixture_lookup_key(home), fixture_lookup_key(away))
            res = results.get(key)
            if not res:
                continue
            actual = res["actual"]
            probs = {o: _f(p.get(f"prob_{OUTCOME_KEY[o]}")) for o in OUTCOMES}
            if any(v is None for v in probs.values()):
                continue
            model_pick = str(p.get("model_pick", ""))
            if model_pick not in OUTCOMES:
                continue
            fair = {o: _f(p.get(f"fair_odds_{OUTCOME_KEY[o]}")) for o in OUTCOMES}
            book = {
                "H": _f(p.get("book_odds_home"), _f(p.get("B365H"))),
                "D": _f(p.get("book_odds_draw"), _f(p.get("B365D"))),
                "A": _f(p.get("book_odds_away"), _f(p.get("B365A"))),
            }
            user_pick = str(picks.get(gw, {}).get(int(p.get("match_idx", i)), ""))
            records.append({
                "gw": gw,
                "date": str(p.get("date", ""))[:10],
                "home_team": home,
                "away_team": away,
                "actual": actual,
                "model_pick": model_pick,
                "user_pick": user_pick if user_pick in OUTCOMES else "",
                "probs": probs,
                "fair_odds": fair,
                "book_odds": book,
            })
    return records, picks


def _roi_fair(r: dict) -> float:
    odds = r["fair_odds"][r["model_pick"]]
    if odds is None or odds <= 1.0:
        return float("nan")
    return odds - 1.0 if r["actual"] == r["model_pick"] else -1.0


def _roi_book(r: dict) -> float:
    odds = r["book_odds"][r["model_pick"]]
    if odds is None or odds <= 1.0:
        return float("nan")
    return odds - 1.0 if r["actual"] == r["model_pick"] else -1.0


def _is_value(r: dict) -> bool:
    """Model probability beats the bookmaker implied probability (margin removed)."""
    odds = r["book_odds"]
    if any(o is None or o <= 1.0 for o in odds.values()):
        return False
    implied = {o: 1.0 / odds[o] for o in OUTCOMES}
    total = sum(implied.values())
    book_p = {o: implied[o] / total for o in OUTCOMES}
    return r["probs"][r["model_pick"]] > book_p[r["model_pick"]] + 0.005


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="write evaluation_report.json")
    args = parser.parse_args()

    records, _picks = _collect()
    if not records:
        print("No settled matches found (predictions_history + results.csv).")
        return

    df = pd.DataFrame(records)

    def _metrics(sub: pd.DataFrame) -> dict[str, Any]:
        if sub.empty:
            return {}
        probs = np.array([[
            sub.iloc[j]["probs"][o] for o in OUTCOMES
        ] for j in range(len(sub))])
        y = np.array([OUTCOMES.index(a) for a in sub["actual"]])
        pred = np.array([OUTCOMES.index(mp) for mp in sub["model_pick"]])
        acc = float(np.mean(pred == y))
        ll = float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None))))
        br = brier_multiclass(probs, y)
        ece, _rows = expected_calibration_error(probs, y)
        fair_roi = float(np.nanmean([_roi_fair(r) for r in sub.to_dict("records")]))
        book_roi = float(np.nanmean([_roi_book(r) for r in sub.to_dict("records")]))
        value = [r for r in sub.to_dict("records") if _is_value(r)]
        value_roi = float(np.nanmean([_roi_book(r) for r in value])) if value else float("nan")
        return {
            "matches": len(sub),
            "accuracy": acc,
            "log_loss": ll,
            "brier": br,
            "ece": ece,
            "roi_fair": fair_roi,
            "roi_book": book_roi,
            "value_bets": len(value),
            "roi_value": value_roi,
        }

    print("=" * 78)
    print("AUGO MODEL EVALUATION  (predictions_history + results.csv)")
    print("=" * 78)
    print(f"{'GW':<5}{'N':>4}{'Acc':>8}{'LogLoss':>9}{'Brier':>8}{'ECE':>8}{'ROI(fair)':>11}{'ROI(book)':>11}")
    for gw in sorted(df["gw"].unique()):
        sub = df[df["gw"] == gw]
        m = _metrics(sub)
        if not m:
            continue
        print(f"{gw:<5}{m['matches']:>4}{m['accuracy']:>8.1%}{m['log_loss']:>9.3f}"
              f"{m['brier']:>8.4f}{m['ece']:>8.3f}{_money(m['roi_fair']):>11}{_money(m['roi_book']):>11}")
    print("-" * 78)
    cum = _metrics(df)
    print(f"{'ALL':<5}{cum['matches']:>4}{cum['accuracy']:>8.1%}{cum['log_loss']:>9.3f}"
          f"{cum['brier']:>8.4f}{cum['ece']:>8.3f}{_money(cum['roi_fair']):>11}{_money(cum['roi_book']):>11}")
    print()
    print(f"Value strategy (model>book implied, bet at book odds): {cum['value_bets']} bets, "
          f"ROI = {_money(cum['roi_value'])}")

    # Calibration table
    probs = np.array([[
        df.iloc[j]["probs"][o] for o in OUTCOMES
    ] for j in range(len(df))])
    y = np.array([OUTCOMES.index(a) for a in df["actual"]])
    _ece, rows = expected_calibration_error(probs, y)
    print("\nCalibration (confidence bins of the model pick):")
    print(f"{'Conf':>8}{'N':>6}{'Hit rate':>10}{'Avg conf':>10}")
    for cnt, acc, avg in rows:
        print(f"{avg:>8.2f}{cnt:>6}{acc:>10.1%}{avg:>10.1%}")

    # User vs model (only where picks exist)
    user = df[df["user_pick"] != ""]
    if not user.empty:
        u_acc = float(np.mean(user["user_pick"] == user["actual"]))
        m_acc = float(np.mean(user["model_pick"] == user["actual"]))
        print(f"\nUser picks: {len(user)} settled  ->  user {u_acc:.1%} vs model {m_acc:.1%}")

    if args.json:
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump({"cumulative": cum, "per_gameweek": {
                str(g): _metrics(df[df["gw"] == g]) for g in sorted(df["gw"].unique())
            }}, f, indent=2)
        print(f"\nReport written -> {REPORT_FILE}")


if __name__ == "__main__":
    main()
