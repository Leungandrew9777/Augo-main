"""dates.py — single source of truth for parsing match dates.

football-data.co.uk alternates between ``DD/MM/YYYY`` and ``DD/MM/YY`` across
seasons. Pandas infers one fixed format from the first row of a mixed frame
and silently coerces the other rows to ``NaT`` — that is how three full
seasons (2013/14, 2014/15, 2016/17) were dropped from training. Parsing with
``format="mixed"`` keeps every row.
"""

from __future__ import annotations

import re
from datetime import date, datetime

import pandas as pd

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def parse_series(values, *, dayfirst: bool = True) -> pd.Series:
    """Parse a column of mixed DD/MM/YYYY and DD/MM/YY values."""
    return pd.to_datetime(values, errors="coerce", dayfirst=dayfirst, format="mixed")


def parse_one(value) -> pd.Timestamp | None:
    """Parse one date value (ISO or UK-style) without day/month ambiguity."""
    raw = str(value).strip() if value is not None else ""
    if not raw:
        return None
    if _ISO_RE.match(raw):
        ts = pd.to_datetime(raw, errors="coerce")
    else:
        ts = pd.to_datetime(raw, errors="coerce", dayfirst=True, format="mixed")
    return ts if pd.notna(ts) else None


def iso_date(value) -> str | None:
    """Return ``YYYY-MM-DD`` for any supported representation, else ``None``."""
    ts = parse_one(value)
    return str(ts.date()) if ts is not None else None


def today_uk() -> date:
    """Today's calendar date in match-local time (Europe/London)."""
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("Europe/London")).date()
    except Exception:
        return datetime.now().date()
