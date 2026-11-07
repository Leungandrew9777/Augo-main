from datetime import date

import pandas as pd

from data_quality import validate_fixtures, validate_results


def _fixtures(rows):
    return pd.DataFrame(rows, columns=["gameweek", "date", "home_team", "away_team"])


def test_mixed_gameweeks_in_round_fails():
    fixtures = _fixtures([
        (9, "31/10/2026", "A", "B"),
        (9, "1/11/2026", "C", "D"),
        (10, "2/11/2026", "E", "F"),
    ])
    errors, _ = validate_fixtures(fixtures, today=date(2026, 10, 30))
    assert any("mixes gameweeks" in message for message in errors)


def test_duplicate_fixture_fails():
    fixtures = _fixtures([
        (9, "31/10/2026", "A", "B"),
        (9, "31/10/2026", "A", "B"),
    ])
    errors, _ = validate_fixtures(fixtures, today=date(2026, 10, 30))
    assert any("duplicate" in message for message in errors)


def test_team_twice_in_a_round_fails():
    fixtures = _fixtures([
        (10, "6/11/2026", "A", "B"),
        (10, "7/11/2026", "A", "C"),
    ])
    errors, _ = validate_fixtures(fixtures, today=date(2026, 10, 30))
    assert any("twice" in message for message in errors)


def test_result_gameweek_mismatch_fails():
    fixtures = _fixtures([(9, "2/11/2026", "Newcastle", "Everton")])
    results = pd.DataFrame([{
        "date": "2/11/2026",
        "gameweek": 10,
        "home_team": "Newcastle",
        "away_team": "Everton",
        "home_goals": 0,
        "away_goals": 1,
    }])
    errors, _ = validate_results(results, fixtures, today=date(2026, 11, 6))
    assert any("GW10" in message and "GW9" in message for message in errors)


def test_unmatched_result_is_warning_only():
    fixtures = _fixtures([(9, "2/11/2026", "Newcastle", "Everton")])
    results = pd.DataFrame([{
        "date": "10/8/2019",
        "gameweek": 1,
        "home_team": "Old",
        "away_team": "Match",
        "home_goals": 1,
        "away_goals": 0,
    }])
    errors, warnings = validate_results(results, fixtures, today=date(2026, 11, 6))
    assert errors == []
    assert any("not found" in message for message in warnings)
