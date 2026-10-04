"""bankroll.py — virtual-wallet strategies, ledgers and portfolio summaries.

Every wallet is defined once in ``WALLETS`` and produces a ledger through
``build_bankroll``. Ledgers are rebuilt from the archived predictions
(``persistence.load_archived_predictions``) + results, so old gameweeks are
never mutated — only recomputed.
"""

from __future__ import annotations

import re
from typing import Any, Callable

OUTCOMES = ("H", "D", "A")
LABELS = {"H": "Home", "D": "Draw", "A": "Away"}

# Wallet metadata for both the summary cards and the detail pages.
# ``ledger_style`` picks the row renderer: "edge" shows edge + Kelly, "flat"
# shows the plain pick/stake row.
WALLETS: list[dict[str, str]] = [
    {
        "id": "model",
        "name": "MODEL",
        "color": "#6C63FF",
        "ledger_style": "flat",
        "method": "Stakes the risk-cap % of the wallet (max 5%) on the model's pick (H/D/A) at its own "
                  "fair odds (1/prob) for every match. A pure test of whether the model beats its own probabilities.",
    },
    {
        "id": "value",
        "name": "VALUE KELLY",
        "color": "#FFB74D",
        "ledger_style": "edge",
        "method": "Kelly-sized stakes (capped by risk cap) on the best positive edge per match — "
                  "model probability above the bookmaker implied probability — placed at bookmaker odds.",
    },
    {
        "id": "edge",
        "name": "TOP-4 EDGES",
        "color": "#9CCC65",
        "ledger_style": "flat",
        "method": "Each gameweek, ranks every match by model-vs-bookmaker probability edge and stakes the "
                  "risk-cap % (Kelly, capped 5%) on the edge outcome of the 4 biggest edges, at bookmaker odds.",
    },
    {
        "id": "elo",
        "name": "TOP-4 ELO",
        "color": "#81D4FA",
        "ledger_style": "flat",
        "method": "Each gameweek, ranks every match by |ELO difference| and stakes the risk-cap % (max 5%) on "
                  "the ELO favourite (home if elo_diff > 0, away if < 0) of the 4 biggest gaps, at fair odds.",
    },
    {
        "id": "draw",
        "name": "DRAW VALUE",
        "color": "#FFC107",
        "ledger_style": "edge",
        "method": "Bets the draw only when the ensemble's draw probability beats the margin-adjusted "
                  "bookmaker draw probability, Kelly-sized (capped by risk cap), at bookmaker odds. "
                  "Draws are the most under-priced outcome, so this isolates that signal.",
    },
    {
        "id": "poisson",
        "name": "POISSON EDGE",
        "color": "#BA68C8",
        "ledger_style": "edge",
        "method": "Each match, ranks H/D/A by the Poisson goal model's probability minus the bookmaker "
                  "implied probability and Kelly-stakes the biggest positive edge at bookmaker odds. "
                  "A second, independent opinion on the same 1X2 market.",
    },
    {
        "id": "over25",
        "name": "OVER 2.5",
        "color": "#4DD0E1",
        "ledger_style": "flat",
        "method": "Bets Over 2.5 total goals whenever the Poisson goal model gives it at least 55%, "
                  "staking the risk cap at the model's fair odds (no bookmaker totals market is stored).",
    },
    {
        "id": "btts",
        "name": "BTTS",
        "color": "#AED581",
        "ledger_style": "flat",
        "method": "Bets Both Teams To Score whenever the Poisson goal model gives it at least 55%, "
                  "staking the risk cap at the model's fair odds.",
    },
    {
        "id": "banker",
        "name": "BANKER",
        "color": "#F06292",
        "ledger_style": "flat",
        "method": "Concentrated: stakes the risk cap on the single highest-confidence model pick of each "
                  "gameweek, at the model's fair odds. Tests whether the top pick is really the best.",
    },
]

_EDGE_STYLE_WALLETS = {w["id"] for w in WALLETS if w["ledger_style"] == "edge"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def format_money(value: Any, *, signed: bool = False) -> str:
    """Format bankroll values as dollars with two decimals."""
    amount = _safe_float(value)
    if signed:
        sign = "+" if amount >= 0 else "-"
        return f"{sign}${abs(amount):.2f}"
    return f"${amount:.2f}"


def _gw_int(label: Any) -> int | None:
    m = re.search(r"GW\s*(\d+)", str(label or ""), re.IGNORECASE)
    return int(m.group(1)) if m else None


def _hda_match_odds(match: dict[str, Any], key: str) -> float | None:
    value = match.get(key)
    return None if value is None else _safe_float(value)


def kelly_fraction(probability: float, decimal_odds: float, risk_cap: float) -> float:
    p = max(0.0, min(1.0, float(probability)))
    odds = float(decimal_odds)
    if odds <= 1.0:
        return 0.0
    b = odds - 1.0
    q = 1.0 - p
    raw = (b * p - q) / b
    return max(0.0, min(float(risk_cap), raw))


def _market_probabilities(match: dict[str, Any]) -> dict[str, float] | None:
    probs = {
        "H": _safe_float(match.get("book_prob_home")),
        "D": _safe_float(match.get("book_prob_draw")),
        "A": _safe_float(match.get("book_prob_away")),
    }
    if all(v > 0 for v in probs.values()):
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

    odds = {
        "H": _safe_float(match.get("book_odds_home")),
        "D": _safe_float(match.get("book_odds_draw")),
        "A": _safe_float(match.get("book_odds_away")),
    }
    if not all(v > 1.0 for v in odds.values()):
        odds = {
            "H": _safe_float(match.get("B365H")),
            "D": _safe_float(match.get("B365D")),
            "A": _safe_float(match.get("B365A")),
        }
    if not all(v > 1.0 for v in odds.values()):
        return None
    implied = {k: 1.0 / v for k, v in odds.items()}
    total = sum(implied.values())
    return {k: v / total for k, v in implied.items()}


def _book_odds(match: dict[str, Any]) -> dict[str, float] | None:
    odds = {
        "H": _safe_float(match.get("book_odds_home")),
        "D": _safe_float(match.get("book_odds_draw")),
        "A": _safe_float(match.get("book_odds_away")),
    }
    if all(v > 1.0 for v in odds.values()):
        return odds
    odds = {
        "H": _safe_float(match.get("B365H")),
        "D": _safe_float(match.get("B365D")),
        "A": _safe_float(match.get("B365A")),
    }
    if all(v > 1.0 for v in odds.values()):
        return odds
    return None


def select_best_edge(match: dict[str, Any]) -> dict[str, Any] | None:
    market = _market_probabilities(match)
    odds = _book_odds(match)
    if market is None or odds is None:
        return None
    model = {
        "H": _safe_float(match.get("prob_home")),
        "D": _safe_float(match.get("prob_draw")),
        "A": _safe_float(match.get("prob_away")),
    }
    candidates = []
    for outcome in OUTCOMES:
        edge = model[outcome] - market[outcome]
        candidates.append({
            "outcome": outcome,
            "p_model": model[outcome],
            "p_market": market[outcome],
            "edge": edge,
            "odds": odds[outcome],
        })
    best = max(candidates, key=lambda x: x["edge"])
    return best if best["edge"] > 0 else None


def _wallet_stake(odds: float, p_model: float, balance: float, risk_cap: float) -> float:
    """% of wallet balance to stake, capped at ``risk_cap``.

    Uses Kelly when the odds are profitable (odds > 1 and a positive fraction),
    otherwise falls back to the full risk cap (e.g. fair-odds wallets where
    Kelly is zero by construction).
    """
    if odds > 1.0:
        f = kelly_fraction(p_model, odds, risk_cap)
        if f > 1e-4:  # ignore floating-point noise (~0 Kelly at fair odds)
            return balance * f
    return balance * risk_cap


def _settle_bet(
    info: dict[str, Any],
    stake: float,
    odds: float,
    result: dict[str, Any],
    grader: Callable[[dict[str, Any], dict[str, Any]], bool | None] | None,
) -> tuple[str, float]:
    """Return (status, pnl). ``grader`` handles non-1X2 markets."""
    if grader is not None:
        verdict = grader(info, result)
        if verdict is None:
            return "pending", 0.0
        return ("win", stake * (odds - 1.0)) if verdict else ("loss", -stake)
    actual = result.get("actual", "")
    if actual not in OUTCOMES:
        return "pending", 0.0
    return ("win", stake * (odds - 1.0)) if actual == info["pick"] else ("loss", -stake)


def _build_wallet_ledger(
    archives: dict[int, dict[str, Any]],
    results: dict[tuple[int, str, str], dict[str, Any]],
    *,
    starting: float,
    risk_cap: float,
    wallet_key: str,
    picker: Callable[[dict[str, Any], int], dict[str, Any] | None],
    top_n: int | None = None,
    grader: Callable[[dict[str, Any], dict[str, Any]], bool | None] | None = None,
) -> list[dict[str, Any]]:
    """Generic per-GW wallet builder: % staking (risk-capped), ranked picks.

    ``picker(match, idx)`` returns a dict with at least:
      pick, odds, p_model, rank (and optional extra fields copied to the bet row).
    ``top_n`` keeps only the ``top_n`` best-ranked picks per gameweek.
    ``grader(info, result)`` settles non-1X2 picks (returns True/False/None).
    """
    ledger: list[dict[str, Any]] = []
    running = starting
    for gw in sorted(archives.keys()):
        cache = archives[gw]
        preds = cache.get("predictions", []) if isinstance(cache, dict) else []
        gw_balance = running
        candidates: list[dict[str, Any]] = []
        for idx, match in enumerate(preds):
            if not isinstance(match, dict):
                continue
            info = picker(match, idx)
            if info:
                info["idx"] = idx
                candidates.append(info)
        if top_n:
            candidates.sort(key=lambda c: -float(c["rank"]))
            candidates = candidates[:top_n]

        gw_rows: list[dict[str, Any]] = []
        for info in candidates:
            pick = info["pick"]
            odds = _safe_float(info["odds"])
            if odds <= 1.0:
                continue
            if grader is None and pick not in OUTCOMES:
                continue
            stake = _wallet_stake(odds, _safe_float(info.get("p_model"), 0.5), gw_balance, risk_cap)
            if stake <= 0:
                continue
            home = str(info["match"].get("home_team", ""))
            away = str(info["match"].get("away_team", ""))
            result = results.get((gw, home, away), {})
            status, pnl = _settle_bet(info, stake, odds, result, grader)
            row = {
                "id": f"GW{gw}-{info['idx']}-{pick}-{wallet_key}",
                "gw": f"GW{gw}",
                "gw_num": gw,
                "date": str(info["match"].get("date", "")),
                "home_team": home,
                "away_team": away,
                "bet_outcome": pick,
                "bet_label": info.get("bet_label", LABELS.get(pick, pick)),
                "odds_source": info.get("odds_source", "fair"),
                "odds": round(odds, 3),
                "stake": round(stake, 2),
                "stake_disp": format_money(stake),
                "stake_pct_disp": f"{stake / gw_balance * 100:.1f}%" if gw_balance > 0 else "—",
                "actual": result.get("actual", ""),
                "status": status,
                "pnl": round(pnl, 2),
                "pnl_disp": format_money(pnl, signed=True) if status != "pending" else "Pending",
            }
            row.update({k: v for k, v in info.items() if k not in
                        ("match", "idx", "pick", "odds", "p_model", "rank") and k not in row})
            gw_rows.append(row)
        ledger.extend(gw_rows)
        running += sum(float(b["pnl"]) for b in gw_rows if b["status"] != "pending")
    return ledger


def build_model_ledger(
    archives: dict[int, dict[str, Any]],
    results: dict[tuple[int, str, str], dict[str, Any]],
    *,
    starting: float = 1000.0,
    risk_cap: float = 0.05,
) -> list[dict[str, Any]]:
    """Wallet A: model pick at its fair odds, every match, risk-capped % stake."""

    def picker(match: dict[str, Any], idx: int) -> dict[str, Any] | None:
        pick = str(match.get("model_pick", ""))
        if pick not in OUTCOMES:
            return None
        odds = {"H": match.get("fair_odds_home"),
                "D": match.get("fair_odds_draw"),
                "A": match.get("fair_odds_away")}.get(pick)
        if not odds or _safe_float(odds) <= 1.0:
            return None
        return {
            "match": match, "pick": pick, "odds": _safe_float(odds),
            "p_model": _safe_float({"H": match.get("prob_home"),
                                    "D": match.get("prob_draw"),
                                    "A": match.get("prob_away")}.get(pick), 0.5),
            "rank": 0.0,
        }

    return _build_wallet_ledger(archives, results, starting=starting, risk_cap=risk_cap,
                                wallet_key="model", picker=picker)


def build_value_ledger(
    archives: dict[int, dict[str, Any]],
    results: dict[tuple[int, str, str], dict[str, Any]],
    *,
    starting: float = 1000.0,
    risk_cap: float = 0.05,
) -> list[dict[str, Any]]:
    """Wallet B: best positive edge per match, Kelly staked and compounded per GW."""

    ledger: list[dict[str, Any]] = []
    current = starting
    for gw in sorted(archives.keys()):
        cache = archives[gw]
        preds = cache.get("predictions", []) if isinstance(cache, dict) else []
        for idx, match in enumerate(preds):
            if not isinstance(match, dict):
                continue
            best = select_best_edge(match)
            if not best:
                continue
            kelly = kelly_fraction(best["p_model"], best["odds"], risk_cap)
            if kelly <= 0:
                continue
            stake = current * kelly
            home = str(match.get("home_team", ""))
            away = str(match.get("away_team", ""))
            result = results.get((gw, home, away), {})
            actual = result.get("actual", "")
            status = "pending"
            pnl = 0.0
            if actual in OUTCOMES:
                if actual == best["outcome"]:
                    status = "win"
                    pnl = stake * (best["odds"] - 1.0)
                else:
                    status = "loss"
                    pnl = -stake
                current += pnl

            ledger.append({
                "id": f"GW{gw}-{idx}-{best['outcome']}",
                "gw": f"GW{gw}",
                "gw_num": gw,
                "date": str(match.get("date", "")),
                "home_team": home,
                "away_team": away,
                "bet_outcome": best["outcome"],
                "bet_label": LABELS.get(best["outcome"], best["outcome"]),
                "p_model": round(best["p_model"], 4),
                "p_model_disp": f"{best['p_model']*100:.1f}%",
                "p_market": round(best["p_market"], 4),
                "edge": round(best["edge"], 4),
                "edge_disp": f"{best['edge']*100:+.1f}pp",
                "odds_source": "book",
                "odds": round(best["odds"], 3),
                "stake": round(stake, 2),
                "stake_disp": format_money(stake),
                "kelly_fraction": round(kelly, 4),
                "kelly_disp": f"{kelly*100:.1f}%",
                "actual": actual,
                "status": status,
                "pnl": round(pnl, 2),
                "pnl_disp": format_money(pnl, signed=True) if status != "pending" else "Pending",
            })
    return ledger


def _pnl_by_gw(ledger: list[dict[str, Any]]) -> dict[int, float]:
    out: dict[int, float] = {}
    for b in ledger:
        if b["status"] == "pending":
            continue
        out[b["gw_num"]] = out.get(b["gw_num"], 0.0) + float(b["pnl"])
    return out


def build_fair_edge_ledger(
    archives: dict[int, dict[str, Any]],
    results: dict[tuple[int, str, str], dict[str, Any]],
    *,
    starting: float = 1000.0,
    risk_cap: float = 0.05,
    top_n: int = 4,
) -> list[dict[str, Any]]:
    """Wallet 3: per GW, the ``top_n`` matches with the biggest model-vs-book edge.

    Risk-capped % stake (Kelly when profitable) on the edge outcome at book odds.
    """

    def picker(match: dict[str, Any], idx: int) -> dict[str, Any] | None:
        best = select_best_edge(match)
        if not best:
            return None
        return {
            "match": match, "pick": best["outcome"], "odds": best["odds"],
            "p_model": best["p_model"], "rank": float(best["edge"]),
            "edge": round(best["edge"], 4),
            "edge_disp": f"{best['edge']*100:+.1f}pp",
            "odds_source": "book",
        }

    return _build_wallet_ledger(archives, results, starting=starting, risk_cap=risk_cap,
                                wallet_key="edge", picker=picker, top_n=top_n)


def build_elo_edge_ledger(
    archives: dict[int, dict[str, Any]],
    results: dict[tuple[int, str, str], dict[str, Any]],
    *,
    starting: float = 1000.0,
    risk_cap: float = 0.05,
    top_n: int = 4,
) -> list[dict[str, Any]]:
    """Wallet 4: per GW, the ``top_n`` matches with the biggest |ELO difference|.

    Risk-capped % stake on the ELO favourite at its fair odds.
    """

    def picker(match: dict[str, Any], idx: int) -> dict[str, Any] | None:
        ed = _safe_float(match.get("elo_diff"))
        pick = "H" if ed > 0 else ("A" if ed < 0 else "")
        if not pick:
            return None
        odds = {"H": match.get("fair_odds_home"), "A": match.get("fair_odds_away")}.get(pick)
        if not odds or _safe_float(odds) <= 1.0:
            return None
        return {
            "match": match, "pick": pick, "odds": _safe_float(odds),
            "p_model": _safe_float({"H": match.get("prob_home"),
                                    "A": match.get("prob_away")}.get(pick), 0.5),
            "rank": abs(ed),
            "elo_diff": round(ed, 0),
            "elo_diff_disp": f"{ed:+.0f}",
        }

    return _build_wallet_ledger(archives, results, starting=starting, risk_cap=risk_cap,
                                wallet_key="elo", picker=picker, top_n=top_n)


def build_draw_value_ledger(
    archives: dict[int, dict[str, Any]],
    results: dict[tuple[int, str, str], dict[str, Any]],
    *,
    starting: float = 1000.0,
    risk_cap: float = 0.05,
    min_prob: float = 0.22,
    min_edge: float = 0.005,
) -> list[dict[str, Any]]:
    """Wallet 5: draw-only value. Ensemble draw probability vs book implied."""

    def picker(match: dict[str, Any], idx: int) -> dict[str, Any] | None:
        market = _market_probabilities(match)
        odds = _book_odds(match)
        p_model = _safe_float(match.get("prob_draw"))
        if market is None or odds is None or p_model < min_prob:
            return None
        edge = p_model - market["D"]
        if edge <= min_edge:
            return None
        kelly = kelly_fraction(p_model, odds["D"], risk_cap)
        return {
            "match": match, "pick": "D", "odds": odds["D"], "p_model": p_model,
            "rank": edge,
            "edge": round(edge, 4),
            "edge_disp": f"{edge*100:+.1f}pp",
            "kelly_fraction": round(kelly, 4),
            "kelly_disp": f"{kelly*100:.1f}%",
            "odds_source": "book",
        }

    return _build_wallet_ledger(archives, results, starting=starting, risk_cap=risk_cap,
                                wallet_key="draw", picker=picker)


def build_poisson_edge_ledger(
    archives: dict[int, dict[str, Any]],
    results: dict[tuple[int, str, str], dict[str, Any]],
    *,
    starting: float = 1000.0,
    risk_cap: float = 0.05,
    min_edge: float = 0.005,
) -> list[dict[str, Any]]:
    """Wallet 6: Poisson model H/D/A vs book implied, biggest positive edge."""

    def picker(match: dict[str, Any], idx: int) -> dict[str, Any] | None:
        market = _market_probabilities(match)
        odds = _book_odds(match)
        if market is None or odds is None:
            return None
        probs = {
            "H": _safe_float(match.get("poisson_prob_home")),
            "D": _safe_float(match.get("poisson_prob_draw")),
            "A": _safe_float(match.get("poisson_prob_away")),
        }
        if sum(probs.values()) < 0.5:
            return None
        edges = {o: probs[o] - market[o] for o in OUTCOMES}
        pick = max(edges, key=lambda o: edges[o])
        if edges[pick] <= min_edge:
            return None
        kelly = kelly_fraction(probs[pick], odds[pick], risk_cap)
        return {
            "match": match, "pick": pick, "odds": odds[pick], "p_model": probs[pick],
            "rank": edges[pick],
            "edge": round(edges[pick], 4),
            "edge_disp": f"{edges[pick]*100:+.1f}pp",
            "kelly_fraction": round(kelly, 4),
            "kelly_disp": f"{kelly*100:.1f}%",
            "odds_source": "book",
        }

    return _build_wallet_ledger(archives, results, starting=starting, risk_cap=risk_cap,
                                wallet_key="poisson", picker=picker)


def _settle_over25(info: dict[str, Any], result: dict[str, Any]) -> bool | None:
    hg, ag = result.get("home_goals"), result.get("away_goals")
    if hg is None or ag is None:
        return None
    return int(hg) + int(ag) > 2.5


def _settle_btts(info: dict[str, Any], result: dict[str, Any]) -> bool | None:
    hg, ag = result.get("home_goals"), result.get("away_goals")
    if hg is None or ag is None:
        return None
    return int(hg) > 0 and int(ag) > 0


def build_over25_ledger(
    archives: dict[int, dict[str, Any]],
    results: dict[tuple[int, str, str], dict[str, Any]],
    *,
    starting: float = 1000.0,
    risk_cap: float = 0.05,
    min_prob: float = 0.55,
) -> list[dict[str, Any]]:
    """Wallet 7: Over 2.5 total goals, model fair odds, risk-capped stake."""

    def picker(match: dict[str, Any], idx: int) -> dict[str, Any] | None:
        p_model = _safe_float(match.get("poisson_over_25"))
        if p_model < min_prob:
            return None
        return {
            "match": match, "pick": "Over 2.5", "bet_label": "Over 2.5",
            "odds": 1.0 / p_model, "p_model": p_model, "rank": p_model,
            "p_model_disp": f"{p_model*100:.1f}%",
            "odds_source": "fair",
        }

    return _build_wallet_ledger(archives, results, starting=starting, risk_cap=risk_cap,
                                wallet_key="over25", picker=picker, grader=_settle_over25)


def build_btts_ledger(
    archives: dict[int, dict[str, Any]],
    results: dict[tuple[int, str, str], dict[str, Any]],
    *,
    starting: float = 1000.0,
    risk_cap: float = 0.05,
    min_prob: float = 0.55,
) -> list[dict[str, Any]]:
    """Wallet 8: Both Teams To Score, model fair odds, risk-capped stake."""

    def picker(match: dict[str, Any], idx: int) -> dict[str, Any] | None:
        p_model = _safe_float(match.get("poisson_btts"))
        if p_model < min_prob:
            return None
        return {
            "match": match, "pick": "BTTS", "bet_label": "BTTS",
            "odds": 1.0 / p_model, "p_model": p_model, "rank": p_model,
            "p_model_disp": f"{p_model*100:.1f}%",
            "odds_source": "fair",
        }

    return _build_wallet_ledger(archives, results, starting=starting, risk_cap=risk_cap,
                                wallet_key="btts", picker=picker, grader=_settle_btts)


def build_banker_ledger(
    archives: dict[int, dict[str, Any]],
    results: dict[tuple[int, str, str], dict[str, Any]],
    *,
    starting: float = 1000.0,
    risk_cap: float = 0.05,
) -> list[dict[str, Any]]:
    """Wallet 9: the single highest-confidence model pick per gameweek, fair odds."""

    def picker(match: dict[str, Any], idx: int) -> dict[str, Any] | None:
        pick = str(match.get("model_pick", ""))
        if pick not in OUTCOMES:
            return None
        odds = {"H": match.get("fair_odds_home"),
                "D": match.get("fair_odds_draw"),
                "A": match.get("fair_odds_away")}.get(pick)
        if not odds or _safe_float(odds) <= 1.0:
            return None
        p_model = _safe_float({"H": match.get("prob_home"),
                               "D": match.get("prob_draw"),
                               "A": match.get("prob_away")}.get(pick), 0.5)
        return {
            "match": match, "pick": pick, "odds": _safe_float(odds),
            "p_model": p_model, "rank": p_model,
            "p_model_disp": f"{p_model*100:.1f}%",
        }

    return _build_wallet_ledger(archives, results, starting=starting, risk_cap=risk_cap,
                                wallet_key="banker", picker=picker, top_n=1)


def build_wallet_ledgers(
    archives: dict[int, dict[str, Any]],
    results: dict[tuple[int, str, str], dict[str, Any]],
    *,
    starting: float = 1000.0,
    risk_cap: float = 0.05,
) -> dict[str, list[dict[str, Any]]]:
    """All wallet ledgers keyed by wallet id (see ``WALLETS``)."""
    build = {
        "model": build_model_ledger,
        "value": build_value_ledger,
        "edge": build_fair_edge_ledger,
        "elo": build_elo_edge_ledger,
        "draw": build_draw_value_ledger,
        "poisson": build_poisson_edge_ledger,
        "over25": build_over25_ledger,
        "btts": build_btts_ledger,
        "banker": build_banker_ledger,
    }
    return {
        wallet["id"]: build[wallet["id"]](archives, results, starting=starting, risk_cap=risk_cap)
        for wallet in WALLETS
    }


def build_pnl_history(ledgers: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Cumulative PnL of every wallet per gameweek (for the comparison chart).

    ``ledgers`` maps a wallet key (e.g. ``"model"``) to its ledger; output rows
    carry ``<key>_pnl`` columns.
    """
    by_gw = {name: _pnl_by_gw(led) for name, led in ledgers.items()}
    gws = sorted(set().union(*[set(v.keys()) for v in by_gw.values()]))
    running = {name: 0.0 for name in ledgers}
    out: list[dict[str, Any]] = []
    for gw in gws:
        row: dict[str, Any] = {"gw": f"GW{gw}"}
        for name in ledgers:
            running[name] += by_gw[name].get(gw, 0.0)
            row[f"{name}_pnl"] = round(running[name], 2)
        out.append(row)
    return out


def _wallet_summary(ledger: list[dict[str, Any]], starting: float) -> dict[str, Any]:
    settled = [b for b in ledger if b["status"] != "pending"]
    pending = [b for b in ledger if b["status"] == "pending"]
    wins = sum(1 for b in settled if b["status"] == "win")
    losses = sum(1 for b in settled if b["status"] == "loss")
    pnl = sum(float(b["pnl"]) for b in settled)
    return {
        "current_bankroll": round(starting + pnl, 2),
        "current_bankroll_disp": format_money(starting + pnl),
        "total_pnl": round(pnl, 2),
        "total_pnl_disp": format_money(pnl, signed=True),
        "pnl_positive": bool(pnl >= 0),
        "roi_disp": f"{(pnl / starting * 100):+.1f}%" if starting > 0 else "—",
        "settled": len(settled),
        "pending": len(pending),
        "wins": wins,
        "losses": losses,
        "record": f"{wins}-{losses}",
    }


def build_bankroll(
    archives: dict[int, dict[str, Any]],
    results: dict[tuple[int, str, str], dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    starting = _safe_float(settings.get("starting_bankroll"), 1000.0)
    risk_cap = max(0.0, min(1.0, _safe_float(settings.get("risk_cap"), 0.05)))

    ledgers = build_wallet_ledgers(archives, results, starting=starting, risk_cap=risk_cap)
    summaries = {wid: _wallet_summary(led, starting) for wid, led in ledgers.items()}
    value = summaries["value"]

    out: dict[str, Any] = {
        # Global settings / aliases for the primary (VALUE KELLY) wallet.
        "starting_bankroll": round(starting, 2),
        "starting_bankroll_disp": format_money(starting),
        "risk_cap": risk_cap,
        "risk_cap_disp": f"{risk_cap*100:.1f}%",
        "current_bankroll": value["current_bankroll"],
        "current_bankroll_disp": value["current_bankroll_disp"],
        "total_pnl": value["total_pnl"],
        "total_pnl_disp": value["total_pnl_disp"],
        "pnl_positive": value["pnl_positive"],
        "roi_disp": value["roi_disp"],
        "record": value["record"],
        "settled_count": value["settled"],
        "pending_count": value["pending"],
        "wins": value["wins"],
        "losses": value["losses"],
        "ledger": ledgers["value"],
        "suggestions": [b for b in ledgers["value"] if b["status"] == "pending"],
        # Per-wallet flattened scalars (used by the UI cards) + latest-20 ledgers.
        "ledgers": {wid: list(reversed(led[-20:])) for wid, led in ledgers.items()},
        "pnl_history": build_pnl_history(ledgers),
    }
    for wid, summary in summaries.items():
        for key, val in summary.items():
            out[f"{wid}_{key}"] = val
    return out


def default_bankroll_summary(starting: float = 1000.0, risk_cap: float = 0.05) -> dict[str, Any]:
    """Placeholder summary used before the first rebuild (mirrors build_bankroll)."""
    zero = {
        "current_bankroll": starting,
        "current_bankroll_disp": format_money(starting),
        "total_pnl": 0.0,
        "total_pnl_disp": "+$0.00",
        "pnl_positive": True,
        "roi_disp": "+0.0%",
        "settled": 0,
        "pending": 0,
        "wins": 0,
        "losses": 0,
        "record": "0-0",
    }
    out: dict[str, Any] = {
        "starting_bankroll": starting,
        "starting_bankroll_disp": format_money(starting),
        "risk_cap": risk_cap,
        "risk_cap_disp": f"{risk_cap*100:.1f}%",
        "current_bankroll": starting,
        "current_bankroll_disp": format_money(starting),
        "total_pnl": 0.0,
        "total_pnl_disp": "+$0.00",
        "pnl_positive": True,
        "roi_disp": "+0.0%",
        "record": "0-0",
        "settled_count": 0,
        "pending_count": 0,
        "wins": 0,
        "losses": 0,
    }
    for wallet in WALLETS:
        for key, val in zero.items():
            out[f"{wallet['id']}_{key}"] = val
    return out


def build_current_suggestions(predictions: list[dict[str, Any]], bankroll: float, risk_cap: float) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    current = max(float(bankroll), 0.0)
    cap = max(0.0, min(1.0, float(risk_cap)))
    for idx, match in enumerate(predictions):
        best = select_best_edge(match)
        if not best:
            continue
        kelly = kelly_fraction(best["p_model"], best["odds"], cap)
        if kelly <= 0:
            continue
        stake = current * kelly
        suggestions.append({
            "id": f"current-{idx}-{best['outcome']}",
            "match": f"{match.get('home_team', '')} vs {match.get('away_team', '')}",
            "bet_outcome": best["outcome"],
            "bet_label": LABELS.get(best["outcome"], best["outcome"]),
            "p_model_disp": f"{best['p_model']*100:.1f}%",
            "edge_disp": f"{best['edge']*100:+.1f}pp",
            "odds": round(best["odds"], 3),
            "stake_disp": format_money(stake),
            "kelly_disp": f"{kelly*100:.1f}%",
        })
    return suggestions
