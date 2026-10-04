"""inference.py — shared serving-time prediction pipeline.

Both the weekly CLI (``run_pipeline.py``) and the Reflex app (``app.py``) build
fixtures -> features -> ensemble probabilities -> Poisson/HT/corner markets
through this module, so the two paths can never drift apart.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from display import format_odds_display
from features import (
    MODEL_FEATURE_COLS,
    detect_model_features,
    ensure_model_features,
)
from league import FALLBACK_BADGE, TEAM_BADGES
from team_aliases import badge_lookup_key

_FACTORIALS = [math.factorial(i) for i in range(21)]


def _append_columns(frame: pd.DataFrame, cols) -> pd.DataFrame:
    """Append/replace columns in one concat (avoids fragmented-frame inserts)."""
    if isinstance(cols, pd.DataFrame):
        cols = {c: cols[c] for c in cols.columns}
    existing = [c for c in cols if c in frame.columns]
    if existing:
        frame = frame.drop(columns=existing)
    return pd.concat([frame, pd.DataFrame(cols, index=frame.index)], axis=1)


def _poisson_pmf(lam: float, max_goals: int) -> list[float]:
    """P(k; lam) for k = 0..max_goals with factorials reused across calls."""
    lam = max(float(lam), 1e-9)
    exp_lam = math.exp(-lam)
    return [exp_lam * (lam ** k) / _FACTORIALS[k] for k in range(max_goals + 1)]


def _scoreline_probs(lambda_home: float, lambda_away: float, max_goals: int, rho: float, over_lines) -> dict:
    """Shared scoreline engine: H/D/A, over-line probs, BTTS, top scores.

    ``over_lines`` is a list of thresholds like [1.5, 2.5, 3.5, 4.5].
    """
    lambda_home = max(float(lambda_home), 0.05)
    lambda_away = max(float(lambda_away), 0.05)

    home_probs = _poisson_pmf(lambda_home, max_goals)
    away_probs = _poisson_pmf(lambda_away, max_goals)

    def tau(h: int, a: int) -> float:
        if h == 0 and a == 0:
            return max(1.0 - lambda_home * lambda_away * rho, 0.01)
        if h == 0 and a == 1:
            return max(1.0 + lambda_home * rho, 0.01)
        if h == 1 and a == 0:
            return max(1.0 + lambda_away * rho, 0.01)
        if h == 1 and a == 1:
            return max(1.0 - rho, 0.01)
        return 1.0

    score_probs: list[dict] = []
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    over = {line: 0.0 for line in over_lines}
    p_btts = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = home_probs[h] * away_probs[a] * tau(h, a)
            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p
            for line in over_lines:
                if h + a > line:
                    over[line] += p
            if h > 0 and a > 0:
                p_btts += p
            score_probs.append({"score": f"{h}-{a}", "p": p})

    # Normalize every market by the same total to absorb the tiny tail beyond
    # max_goals (and the Dixon-Coles tau reweighting). H/D/A, over-lines, BTTS
    # and correct-score probabilities then all come from one distribution.
    total = p_home + p_draw + p_away
    if total > 0:
        p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total
        over = {line: mass / total for line, mass in over.items()}
        p_btts /= total
        score_probs = [{"score": s["score"], "p": s["p"] / total} for s in score_probs]

    top_scores = sorted(score_probs, key=lambda x: x["p"], reverse=True)
    return {
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "over": over, "p_btts": p_btts, "top_scores": top_scores,
    }


def poisson_markets(lambda_home: float, lambda_away: float, *, max_goals: int = 6, top_k: int = 5, rho: float = -0.12) -> dict:
    """Full-time Poisson markets with a Dixon-Coles low-score correction.

    ``rho`` (< 0) boosts 0-0 / 1-0 / 0-1 / 1-1 scorelines (draw-heavy results).
    Returns O1.5 / O2.5 / O3.5 / O4.5 total-goal probabilities plus BTTS.
    """
    res = _scoreline_probs(lambda_home, lambda_away, max_goals, rho, [1.5, 2.5, 3.5, 4.5])
    return {
        "poisson_prob_home": res["p_home"],
        "poisson_prob_draw": res["p_draw"],
        "poisson_prob_away": res["p_away"],
        "poisson_over_15": res["over"][1.5],
        "poisson_over_25": res["over"][2.5],
        "poisson_over_35": res["over"][3.5],
        "poisson_over_45": res["over"][4.5],
        "poisson_btts": res["p_btts"],
        "poisson_correct_scores": [
            {
                "score": s["score"],
                "p": round(float(s["p"]), 6),
                "disp_p": f"{float(s['p']) * 100:.1f}%",
            }
            for s in res["top_scores"][:top_k]
        ],
    }


def ht_markets(lambda_home: float, lambda_away: float, *, max_goals: int = 4, rho: float = -0.15) -> dict:
    """Half-time markets: HT H/D/A + HT O0.5/O1.5/O2.5 goals (Dixon-Coles)."""
    res = _scoreline_probs(lambda_home, lambda_away, max_goals, rho, [0.5, 1.5, 2.5])
    return {
        "ht_prob_home": res["p_home"],
        "ht_prob_draw": res["p_draw"],
        "ht_prob_away": res["p_away"],
        "ht_over_05": res["over"][0.5],
        "ht_over_15": res["over"][1.5],
        "ht_over_25": res["over"][2.5],
    }


def corner_markets(total_lambda: float) -> dict:
    """O/U total-corners probabilities from a Poisson on the summed lambda."""
    total_lambda = max(float(total_lambda), 0.5)
    max_line = 12
    tail = 0.0
    cdf_at: dict[int, float] = {}
    for k, p in enumerate(_poisson_pmf(total_lambda, max_line)):
        tail += p
        cdf_at[k] = tail

    out: dict[str, float] = {}
    for line in (8.5, 9.5, 10.5, 11.5, 12.5):
        out[f"corner_over_{str(line).replace('.', '')}"] = 1.0 - cdf_at[int(math.floor(line))]
    return out


def _disp_poisson_vs_ensemble_row(r: pd.Series) -> str:
    """Poisson minus ensemble outcome probabilities, in percentage points."""
    try:
        e_h, e_d, e_a = float(r["prob_home"]), float(r["prob_draw"]), float(r["prob_away"])
        p_h, p_d, p_a = (
            float(r["poisson_prob_home"]),
            float(r["poisson_prob_draw"]),
            float(r["poisson_prob_away"]),
        )
    except (TypeError, ValueError, KeyError):
        return "—"
    d_h = (p_h - e_h) * 100.0
    d_d = (p_d - e_d) * 100.0
    d_a = (p_a - e_a) * 100.0
    return f"Poisson−ens. (pp): H {d_h:+.0f} · D {d_d:+.0f} · A {d_a:+.0f}"


def predict_matches(upcoming: pd.DataFrame, model, df_elo: pd.DataFrame) -> pd.DataFrame:
    """Ensemble probabilities plus every display field for upcoming fixtures.

    Model classes are encoded as 0=Away, 1=Draw, 2=Home.
    """
    feature_cols = detect_model_features(model) or MODEL_FEATURE_COLS
    upcoming = ensure_model_features(upcoming, df_elo, feature_cols)
    probs = model.predict_proba(upcoming[feature_cols])

    base_cols = {
        "prob_away": probs[:, 0],
        "prob_draw": probs[:, 1],
        "prob_home": probs[:, 2],
    }
    base_cols["fair_odds_home"] = 1.0 / base_cols["prob_home"]
    base_cols["fair_odds_draw"] = 1.0 / base_cols["prob_draw"]
    base_cols["fair_odds_away"] = 1.0 / base_cols["prob_away"]
    upcoming = _append_columns(upcoming, base_cols)

    disp: dict[str, Any] = {
        "disp_odds_home": upcoming["fair_odds_home"].map(format_odds_display),
        "disp_odds_draw": upcoming["fair_odds_draw"].map(format_odds_display),
        "disp_odds_away": upcoming["fair_odds_away"].map(format_odds_display),
        "disp_prob_home": upcoming["prob_home"].map(lambda v: f"{v*100:.1f}%"),
        "disp_prob_draw": upcoming["prob_draw"].map(lambda v: f"{v*100:.1f}%"),
        "disp_prob_away": upcoming["prob_away"].map(lambda v: f"{v*100:.1f}%"),
        "disp_elo_diff": upcoming["elo_diff"].map(lambda v: f"{v:+.0f}"),
        "badge_home": upcoming["home_team"].map(
            lambda t: TEAM_BADGES.get(badge_lookup_key(str(t)), FALLBACK_BADGE)
        ),
        "badge_away": upcoming["away_team"].map(
            lambda t: TEAM_BADGES.get(badge_lookup_key(str(t)), FALLBACK_BADGE)
        ),
        "chart_label": (
            upcoming["home_team"].astype(str).str[:3].str.upper()
            + " v "
            + upcoming["away_team"].astype(str).str[:3].str.upper()
        ),
        "model_pick": np.array(["A", "D", "H"])[np.argmax(probs, axis=1)],
    }

    # Optional bookmaker odds (from fixtures.csv if provided).
    if all(c in upcoming.columns for c in ("B365H", "B365D", "B365A")):
        book_h = pd.to_numeric(upcoming["B365H"], errors="coerce")
        book_d = pd.to_numeric(upcoming["B365D"], errors="coerce")
        book_a = pd.to_numeric(upcoming["B365A"], errors="coerce")
        valid = (book_h > 0) & (book_d > 0) & (book_a > 0)

        book_odds_home = book_h.where(valid, np.nan)
        book_odds_draw = book_d.where(valid, np.nan)
        book_odds_away = book_a.where(valid, np.nan)

        inv_h = (1.0 / book_h).where(valid, np.nan)
        inv_d = (1.0 / book_d).where(valid, np.nan)
        inv_a = (1.0 / book_a).where(valid, np.nan)
        total = (inv_h + inv_d + inv_a).where(valid, np.nan)

        book_prob_home = (inv_h / total).where(valid, np.nan)
        book_prob_draw = (inv_d / total).where(valid, np.nan)
        book_prob_away = (inv_a / total).where(valid, np.nan)

        def _disp_pct(v):
            return f"{v*100:.1f}%" if pd.notna(v) else ""

        disp.update({
            "book_odds_home": book_odds_home,
            "book_odds_draw": book_odds_draw,
            "book_odds_away": book_odds_away,
            "book_prob_home": book_prob_home,
            "book_prob_draw": book_prob_draw,
            "book_prob_away": book_prob_away,
            "disp_book_odds_home": book_odds_home.map(
                lambda v: format_odds_display(v) if pd.notna(v) else ""),
            "disp_book_odds_draw": book_odds_draw.map(
                lambda v: format_odds_display(v) if pd.notna(v) else ""),
            "disp_book_odds_away": book_odds_away.map(
                lambda v: format_odds_display(v) if pd.notna(v) else ""),
            "disp_book_prob_home": book_prob_home.map(_disp_pct),
            "disp_book_prob_draw": book_prob_draw.map(_disp_pct),
            "disp_book_prob_away": book_prob_away.map(_disp_pct),
        })
    else:
        for col in (
            "book_odds_home", "book_odds_draw", "book_odds_away",
            "book_prob_home", "book_prob_draw", "book_prob_away",
        ):
            disp[col] = pd.NA
        for col in (
            "disp_book_odds_home", "disp_book_odds_draw", "disp_book_odds_away",
            "disp_book_prob_home", "disp_book_prob_draw", "disp_book_prob_away",
        ):
            disp[col] = ""

    return _append_columns(upcoming, disp)


def add_poisson_outputs(upcoming: pd.DataFrame, home_goal_model, away_goal_model, df_elo: pd.DataFrame) -> pd.DataFrame:
    """Full-time goal markets: per-side lambda + Poisson H/D/A, O/U, BTTS."""
    home_features = detect_model_features(home_goal_model)
    away_features = detect_model_features(away_goal_model)
    if not home_features or not away_features:
        return upcoming

    all_features = sorted(set(home_features) | set(away_features))
    upcoming = ensure_model_features(upcoming, df_elo, all_features)
    lambda_home = home_goal_model.predict(upcoming[home_features])
    lambda_away = away_goal_model.predict(upcoming[away_features])

    base_cols = {
        "lambda_home": np.clip(np.asarray(lambda_home, dtype=float), 0.05, None),
        "lambda_away": np.clip(np.asarray(lambda_away, dtype=float), 0.05, None),
    }
    upcoming = _append_columns(upcoming, base_cols)
    markets = upcoming.apply(
        lambda r: poisson_markets(float(r["lambda_home"]), float(r["lambda_away"])),
        axis=1,
    )
    upcoming = _append_columns(upcoming, pd.DataFrame(list(markets), index=upcoming.index))

    tot_lambda = upcoming["lambda_home"] + upcoming["lambda_away"]
    disp: dict[str, pd.Series] = {
        "disp_lambda_home": upcoming["lambda_home"].map(lambda v: f"{v:.2f}"),
        "disp_lambda_away": upcoming["lambda_away"].map(lambda v: f"{v:.2f}"),
        "disp_poisson_prob_home": upcoming["poisson_prob_home"].map(lambda v: f"{v*100:.1f}%"),
        "disp_poisson_prob_draw": upcoming["poisson_prob_draw"].map(lambda v: f"{v*100:.1f}%"),
        "disp_poisson_prob_away": upcoming["poisson_prob_away"].map(lambda v: f"{v*100:.1f}%"),
        "disp_poisson_xg_total": tot_lambda.map(lambda v: f"{float(v):.2f}"),
    }
    for line, name in ((1.5, "disp_poisson_o15"), (2.5, "disp_poisson_o25"),
                       (3.5, "disp_poisson_o35"), (4.5, "disp_poisson_o45")):
        disp[name] = upcoming[f"poisson_over_{str(line).replace('.', '')}"].map(
            lambda v: f"{float(v) * 100:.1f}%")
    disp["disp_poisson_btts"] = upcoming["poisson_btts"].map(lambda v: f"{float(v) * 100:.1f}%")
    disp["disp_poisson_vs_ensemble"] = upcoming.apply(_disp_poisson_vs_ensemble_row, axis=1)
    return _append_columns(upcoming, disp)


def add_ht_outputs(upcoming: pd.DataFrame, ht_home_model, ht_away_model, df_elo: pd.DataFrame) -> pd.DataFrame:
    """Half-time markets: HT H/D/A + HT O0.5/O1.5/O2.5 goals."""
    home_features = detect_model_features(ht_home_model)
    away_features = detect_model_features(ht_away_model)
    if not home_features or not away_features:
        return upcoming
    all_features = sorted(set(home_features) | set(away_features))
    upcoming = ensure_model_features(upcoming, df_elo, all_features)
    base_cols = {
        "lambda_ht_home": np.clip(np.asarray(ht_home_model.predict(upcoming[home_features]), dtype=float), 0.05, None),
        "lambda_ht_away": np.clip(np.asarray(ht_away_model.predict(upcoming[away_features]), dtype=float), 0.05, None),
    }
    upcoming = _append_columns(upcoming, base_cols)
    markets = upcoming.apply(
        lambda r: ht_markets(float(r["lambda_ht_home"]), float(r["lambda_ht_away"])), axis=1)
    upcoming = _append_columns(upcoming, pd.DataFrame(list(markets), index=upcoming.index))

    disp: dict[str, pd.Series] = {
        "disp_lambda_ht_home": upcoming["lambda_ht_home"].map(lambda v: f"{v:.2f}"),
        "disp_lambda_ht_away": upcoming["lambda_ht_away"].map(lambda v: f"{v:.2f}"),
        "disp_ht_prob_home": upcoming["ht_prob_home"].map(lambda v: f"{v*100:.1f}%"),
        "disp_ht_prob_draw": upcoming["ht_prob_draw"].map(lambda v: f"{v*100:.1f}%"),
        "disp_ht_prob_away": upcoming["ht_prob_away"].map(lambda v: f"{v*100:.1f}%"),
        "disp_ht_o05": upcoming["ht_over_05"].map(lambda v: f"{float(v)*100:.1f}%"),
        "disp_ht_o15": upcoming["ht_over_15"].map(lambda v: f"{float(v)*100:.1f}%"),
        "disp_ht_o25": upcoming["ht_over_25"].map(lambda v: f"{float(v)*100:.1f}%"),
    }
    return _append_columns(upcoming, disp)


def add_corner_outputs(upcoming: pd.DataFrame, corner_home_model, corner_away_model, df_elo: pd.DataFrame) -> pd.DataFrame:
    """Total-corners markets: per-side lambda + O8.5..O12.5 probabilities."""
    home_features = detect_model_features(corner_home_model)
    away_features = detect_model_features(corner_away_model)
    if not home_features or not away_features:
        return upcoming
    all_features = sorted(set(home_features) | set(away_features))
    upcoming = ensure_model_features(upcoming, df_elo, all_features)
    corner_home = np.clip(np.asarray(corner_home_model.predict(upcoming[home_features]), dtype=float), 0.5, None)
    corner_away = np.clip(np.asarray(corner_away_model.predict(upcoming[away_features]), dtype=float), 0.5, None)
    base_cols = {
        "corner_home": corner_home,
        "corner_away": corner_away,
        "corner_total": corner_home + corner_away,
    }
    upcoming = _append_columns(upcoming, base_cols)
    markets = upcoming.apply(lambda r: corner_markets(float(r["corner_total"])), axis=1)
    upcoming = _append_columns(upcoming, pd.DataFrame(list(markets), index=upcoming.index))

    disp: dict[str, pd.Series] = {
        "disp_corner_total": upcoming["corner_total"].map(lambda v: f"{float(v):.1f}"),
    }
    for line in (8.5, 9.5, 10.5, 11.5, 12.5):
        key = f"corner_over_{str(line).replace('.', '')}"
        disp[f"disp_{key}"] = upcoming[key].map(lambda v: f"{float(v)*100:.1f}%")
    return _append_columns(upcoming, disp)
