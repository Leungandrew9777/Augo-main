from datetime import date

import pandas as pd
import pytest

from dates import iso_date, parse_series
from rounds import cluster_dated, renumber_rounds


def _rows(*specs):
    return [
        {"gameweek": gw, "date": d, "home_team": f"H{i}", "away_team": f"A{i}"}
        for i, (gw, d) in enumerate(specs)
    ]


def test_parse_series_keeps_two_digit_years():
    parsed = parse_series(pd.Series(["21/08/2026", "17/08/13", "16/08/14"]))
    assert parsed.notna().all()
    assert parsed.dt.strftime("%Y-%m-%d").tolist() == ["2026-08-21", "2013-08-17", "2014-08-16"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-09-12", "2026-09-12"),
        ("12/9/2026", "2026-09-12"),
        ("12/09/13", "2013-09-12"),
        ("not a date", None),
    ],
)
def test_iso_date(value, expected):
    assert iso_date(value) == expected


def test_renumber_merges_late_monday_fixture():
    rows = _rows(
        (9, "31/10/2026"),
        (9, "1/11/2026"),
        (10, "2/11/2026"),  # Monday game appended as its own gameweek
        (11, "6/11/2026"),
        (12, "7/11/2026"),
        (12, "8/11/2026"),
    )
    renumber_rounds(rows, today=date(2026, 11, 6))
    assert [r["gameweek"] for r in rows] == [9, 9, 9, 10, 10, 10]


def test_renumber_renumbers_future_rounds_sequentially():
    rows = _rows(
        (8, "23/10/2026"),
        (8, "24/10/2026"),
        (9, "31/10/2026"),
        (11, "21/11/2026"),
        (12, "22/11/2026"),
        (12, "23/11/2026"),
        (14, "27/11/2026"),
    )
    renumber_rounds(rows, today=date(2026, 10, 26))
    assert [r["gameweek"] for r in rows] == [8, 8, 9, 10, 10, 10, 11]


def test_renumber_keeps_started_round_labels():
    rows = _rows(
        (5, "19/9/2026"),
        (5, "20/9/2026"),
        (6, "10/10/2026"),
    )
    renumber_rounds(rows, today=date(2026, 10, 12))
    assert [r["gameweek"] for r in rows] == [5, 5, 6]


def test_cluster_dated_groups_contiguous_days():
    rows = _rows(
        (1, "2026-10-31"),
        (1, "2026-11-01"),
        (1, "2026-11-02"),
        (2, "2026-11-06"),
    )
    clusters = cluster_dated(rows)
    assert [len(cluster) for cluster in clusters] == [3, 1]
