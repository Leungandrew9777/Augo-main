from __future__ import annotations

from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _driver(text: str, score: float, direction: str) -> dict[str, Any]:
    return {
        "text": text,
        "score": round(float(score), 3),
        "direction": direction,
        "strength": min(abs(float(score)), 1.0),
    }


def build_explanation(row: dict[str, Any]) -> dict[str, Any]:
    """Build simple, rule-based reasons for the model lean."""
    home = str(row.get("home_team", "Home"))
    away = str(row.get("away_team", "Away"))
    model_pick = str(row.get("model_pick", ""))
    probs = {
        "H": _safe_float(row.get("prob_home")),
        "D": _safe_float(row.get("prob_draw")),
        "A": _safe_float(row.get("prob_away")),
    }
    model_prob = probs.get(model_pick, max(probs.values() or [0.0]))

    raw_drivers: list[dict[str, Any]] = []

    elo_diff = _safe_float(row.get("elo_diff"))
    if abs(elo_diff) >= 35:
        side = home if elo_diff > 0 else away
        raw_drivers.append(_driver(f"{side} has the ELO edge ({elo_diff:+.0f}).", elo_diff / 180.0, "H" if elo_diff > 0 else "A"))

    xg_diff = _safe_float(row.get("diff_avg_xG", row.get("xg_diff")))
    if abs(xg_diff) >= 0.15:
        side = home if xg_diff > 0 else away
        raw_drivers.append(_driver(f"{side} has created better recent chances (xG diff {xg_diff:+.2f}).", xg_diff / 0.8, "H" if xg_diff > 0 else "A"))

    xga_diff = _safe_float(row.get("diff_avg_xGA"))
    if abs(xga_diff) >= 0.15:
        side = home if xga_diff < 0 else away
        raw_drivers.append(_driver(f"{side} has the stronger recent defensive xGA profile ({xga_diff:+.2f}).", -xga_diff / 0.8, "H" if xga_diff < 0 else "A"))

    form_diff = _safe_float(row.get("diff_Form"))
    if abs(form_diff) >= 0.25:
        side = home if form_diff > 0 else away
        raw_drivers.append(_driver(f"{side} has better recent form ({form_diff:+.2f} pts/match).", form_diff / 1.5, "H" if form_diff > 0 else "A"))

    h2h_home = _safe_float(row.get("h2h_home_wins"), 0.5)
    if abs(h2h_home - 0.5) >= 0.2:
        side = home if h2h_home > 0.5 else away
        raw_drivers.append(_driver(f"Recent H2H leans toward {side} ({h2h_home*100:.0f}% home-perspective win rate).", (h2h_home - 0.5) * 1.5, "H" if h2h_home > 0.5 else "A"))

    draw_prob = probs.get("D", 0.0)
    if draw_prob >= 0.28:
        raw_drivers.append(_driver(f"Draw is live at {draw_prob*100:.1f}% in the model.", draw_prob, "D"))

    raw_drivers = sorted(raw_drivers, key=lambda d: abs(float(d["score"])), reverse=True)
    drivers = raw_drivers[:5]

    notes: list[str] = []
    if model_prob < 0.45:
        notes.append("Low-confidence lean; probabilities are relatively tight.")
    if not drivers:
        notes.append("No single driver stands out; this looks balanced by the available features.")

    return {
        "pick": model_pick,
        "pick_prob": model_prob,
        "pick_prob_disp": f"{model_prob*100:.1f}%",
        "drivers": drivers,
        "driver_summary": " | ".join(d["text"] for d in drivers[:3]) if drivers else "No major driver detected.",
        "notes": notes,
        "notes_summary": " | ".join(notes),
    }
