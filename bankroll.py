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


def _build_wallet_ledger(
    archives: dict[int, dict[str, Any]],
    results: dict[tuple[int, str, str], dict[str, Any]],
    *,
    starting: float,
    risk_cap: float,
    wallet_key: str,
    picker,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Generic per-GW wallet builder: % staking (risk-capped), ranked picks.

    ``picker(match, idx)`` returns a dict with at least:
      pick, odds, p_model, rank (and optional extra fields copied to the bet row).
    ``top_n`` keeps only the ``top_n`` best-ranked picks per gameweek.
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
            if pick not in OUTCOMES or odds <= 1.0:
                continue
            stake = _wallet_stake(odds, _safe_float(info.get("p_model"), 0.5), gw_balance, risk_cap)
            if stake <= 0:
                continue
            home = str(info["match"].get("home_team", ""))
            away = str(info["match"].get("away_team", ""))
            actual = results.get((gw, home, away), {}).get("actual", "")
            status = "pending"
            pnl = 0.0
            if actual in OUTCOMES:
                if actual == pick:
                    status = "win"
                    pnl = stake * (odds - 1.0)
                else:
                    status = "loss"
                    pnl = -stake
            row = {
                "id": f"GW{gw}-{info['idx']}-{pick}-{wallet_key}",
                "gw": f"GW{gw}",
                "gw_num": gw,
                "date": str(info["match"].get("date", "")),
                "home_team": home,
                "away_team": away,
                "bet_outcome": pick,
                "bet_label": LABELS.get(pick, pick),
                "odds_source": info.get("odds_source", "fair"),
                "odds": round(odds, 3),
                "stake": round(stake, 2),
                "stake_disp": format_money(stake),
                "stake_pct_disp": f"{stake / gw_balance * 100:.1f}%",
                "actual": actual,
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


def build_bankroll(
    archives: dict[int, dict[str, Any]],
    results: dict[tuple[int, str, str], dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    starting = _safe_float(settings.get("starting_bankroll"), 1000.0)
    risk_cap = max(0.0, min(1.0, _safe_float(settings.get("risk_cap"), 0.05)))
    current = starting
    ledger: list[dict[str, Any]] = []
    model_ledger = build_model_ledger(archives, results, starting=starting, risk_cap=risk_cap)
    edge_ledger = build_fair_edge_ledger(archives, results, starting=starting, risk_cap=risk_cap)
    elo_ledger = build_elo_edge_ledger(archives, results, starting=starting, risk_cap=risk_cap)

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

    m_settled = [b for b in model_ledger if b["status"] != "pending"]
    m_wins = sum(1 for b in m_settled if b["status"] == "win")
    m_losses = sum(1 for b in m_settled if b["status"] == "loss")
    m_pnl = sum(float(b["pnl"]) for b in m_settled)
    m_current = starting + m_pnl

    def _wallet_summary(led: list[dict[str, Any]]) -> dict[str, Any]:
        settled = [b for b in led if b["status"] != "pending"]
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
            "record": f"{wins}-{losses}",
        }

    edge_sum = _wallet_summary(edge_ledger)
    elo_sum = _wallet_summary(elo_ledger)

    return {
        # ── Value wallet (Kelly staking at book odds) ──
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
        # ── Model wallet (flat 1u at fair odds) ──
        "model_current_bankroll": round(m_current, 2),
        "model_current_bankroll_disp": format_money(m_current),
        "model_total_pnl": round(m_pnl, 2),
        "model_total_pnl_disp": format_money(m_pnl, signed=True),
        "model_pnl_positive": bool(m_pnl >= 0),
        "model_roi_disp": f"{(m_pnl / starting * 100):+.1f}%" if starting > 0 else "—",
        "model_settled": len(m_settled),
        "model_record": f"{m_wins}-{m_losses}",
        "model_ledger": model_ledger,
        "latest_model_ledger": list(reversed(model_ledger[-20:])),
        # ── Wallet 3: biggest fair edges ──
        "edge_current_bankroll": edge_sum["current_bankroll"],
        "edge_current_bankroll_disp": edge_sum["current_bankroll_disp"],
        "edge_total_pnl": edge_sum["total_pnl"],
        "edge_total_pnl_disp": edge_sum["total_pnl_disp"],
        "edge_pnl_positive": edge_sum["pnl_positive"],
        "edge_roi_disp": edge_sum["roi_disp"],
        "edge_settled": edge_sum["settled"],
        "edge_record": edge_sum["record"],
        "edge_ledger": edge_ledger,
        "latest_edge_ledger": list(reversed(edge_ledger[-20:])),
        # ── Wallet 4: biggest ELO differences ──
        "elo_current_bankroll": elo_sum["current_bankroll"],
        "elo_current_bankroll_disp": elo_sum["current_bankroll_disp"],
        "elo_total_pnl": elo_sum["total_pnl"],
        "elo_total_pnl_disp": elo_sum["total_pnl_disp"],
        "elo_pnl_positive": elo_sum["pnl_positive"],
        "elo_roi_disp": elo_sum["roi_disp"],
        "elo_settled": elo_sum["settled"],
        "elo_record": elo_sum["record"],
        "elo_ledger": elo_ledger,
        "latest_elo_ledger": list(reversed(elo_ledger[-20:])),
        # ── Comparison history (all wallets, cumulative per GW) ──
        "pnl_history": build_pnl_history({
            "model": model_ledger,
            "value": ledger,
            "edge": edge_ledger,
            "elo": elo_ledger,
        }),
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
