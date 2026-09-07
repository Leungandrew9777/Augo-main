from __future__ import annotations

import re
from typing import Any


OUTCOMES = ("H", "D", "A")
LABELS = {"H": "Home", "D": "Draw", "A": "Away"}


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
    m = re.search(r"\d+", str(label or ""))
    return int(m.group(0)) if m else None


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


def build_bankroll(
    archives: dict[int, dict[str, Any]],
    results: dict[tuple[int, str, str], dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    starting = _safe_float(settings.get("starting_bankroll"), 1000.0)
    risk_cap = max(0.0, min(1.0, _safe_float(settings.get("risk_cap"), 0.05)))
    current = starting
    ledger: list[dict[str, Any]] = []

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
            key = (gw, home, away)
            actual = results.get(key, {}).get("actual", "")
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

    settled = [b for b in ledger if b["status"] != "pending"]
    pending = [b for b in ledger if b["status"] == "pending"]
    wins = sum(1 for b in settled if b["status"] == "win")
    losses = sum(1 for b in settled if b["status"] == "loss")
    total_pnl = current - starting
    return {
        "starting_bankroll": round(starting, 2),
        "starting_bankroll_disp": format_money(starting),
        "current_bankroll": round(current, 2),
        "current_bankroll_disp": format_money(current),
        "risk_cap": risk_cap,
        "risk_cap_disp": f"{risk_cap*100:.1f}%",
        "total_pnl": round(total_pnl, 2),
        "total_pnl_disp": format_money(total_pnl, signed=True),
        "pnl_positive": bool(total_pnl >= 0),
        "roi_disp": f"{(total_pnl / starting * 100):+.1f}%" if starting > 0 else "—",
        "settled_count": len(settled),
        "pending_count": len(pending),
        "wins": wins,
        "losses": losses,
        "record": f"{wins}-{losses}",
        "ledger": ledger,
        "latest_ledger": list(reversed(ledger[-20:])),
        "suggestions": pending,
    }


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
