"""display.py — shared display formatting helpers.

Kept dependency-free so both the CLI pipeline (run_pipeline.py) and the Reflex
UI (app.py) render the same strings from the same cache fields.
"""

from __future__ import annotations

import math

ODDS_DISPLAY_CAP = 200.0


def format_odds_display(v, cap: float = ODDS_DISPLAY_CAP) -> str:
    """Format decimal odds for UI/cache display without scientific notation."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return f"{cap:.0f}"

    if not math.isfinite(x):
        return f"{cap:.0f}"

    if x >= cap:
        return f"{cap:.0f}"
    if x >= 10:
        return f"{x:.1f}".rstrip("0").rstrip(".")
    if x >= 1:
        return f"{x:.2f}".rstrip("0").rstrip(".")
    return f"{x:.3f}".rstrip("0").rstrip(".")
