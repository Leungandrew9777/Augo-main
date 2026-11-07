"""rounds.py — gameweek (round) grouping derived from fixture dates.

A round is a block of consecutive fixture dates; a gap of more than one day
starts a new round. Rounds that have already started keep their existing
number and rounds entirely in the future are renumbered sequentially, which
keeps ``fixtures.csv`` aligned with the official Premier League calendar when
fixtures are announced in batches (e.g. a Monday game added later must join
its round instead of becoming a new gameweek).
"""

from __future__ import annotations

from datetime import date, datetime

from dates import parse_one, today_uk


def cluster_dated(rows: list[dict], *, date_key: str = "date") -> list[list[tuple[date, dict]]]:
    """Group rows into date-contiguous clusters (rounds), sorted by date."""
    dated: list[tuple[date, dict]] = []
    for row in rows:
        ts = parse_one(row.get(date_key))
        if ts is None:
            continue
        dated.append((ts.date(), row))
    dated.sort(key=lambda item: item[0])

    clusters: list[list[tuple[date, dict]]] = []
    cur: list[tuple[date, dict]] = []
    prev: date | None = None
    for d, row in dated:
        if prev is not None and (d - prev).days > 1:
            clusters.append(cur)
            cur = []
        cur.append((d, row))
        prev = d
    if cur:
        clusters.append(cur)
    return clusters


def renumber_rounds(rows: list[dict], *, today: date | None = None) -> list[dict]:
    """Renumber gameweeks from the fixture dates, in place.

    Already-started rounds keep their lowest existing number (so results and
    history never move); entirely-future rounds are numbered sequentially
    after the last started round.
    """
    if not rows:
        return rows
    today = today or today_uk()
    clusters = cluster_dated(rows)

    def _gw(row: dict) -> int | None:
        try:
            return int(float(row.get("gameweek")))
        except (TypeError, ValueError):
            return None

    last_gw = 0
    changed = False
    for cluster in clusters:
        labels = [g for g in (_gw(r) for _, r in cluster) if g is not None]
        started = any(d < today for d, _ in cluster)
        if started:
            # A round under way keeps its number; fixtures appended to the same
            # date cluster inherit the round's lowest (original) number.
            gw = min(labels) if labels else last_gw + 1
        else:
            gw = last_gw + 1
        for _, row in cluster:
            if _gw(row) != gw:
                row["gameweek"] = gw
                changed = True
        last_gw = gw

    if changed:
        print("   Renumbered gameweeks from fixture dates to match the official rounds.")
    return rows
