import pandas as pd

from data_pipeline import DataCleaner
from data_quality import validate_clean_history


def test_clean_keeps_two_digit_year_dates():
    raw = pd.DataFrame([
        {"Date": "21/08/2026", "HomeTeam": "A", "AwayTeam": "B", "FTR": "H"},
        {"Date": "17/08/13", "HomeTeam": "C", "AwayTeam": "D", "FTR": "D"},
        {"Date": "16/08/14", "HomeTeam": "E", "AwayTeam": "F", "FTR": "A"},
    ])
    clean = DataCleaner.clean(raw)
    assert len(clean) == 3
    assert clean["Date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2013-08-17", "2014-08-16", "2026-08-21",
    ]


def test_clean_history_validation_flags_missing_season():
    frame = pd.DataFrame([
        {"Date": "2026-08-21", "HomeTeam": "A", "AwayTeam": "B", "FTR": "H", "Season": "2627"},
        {"Date": "2025-08-15", "HomeTeam": "C", "AwayTeam": "D", "FTR": "A", "Season": "2526"},
        {"Date": "2025-08-16", "HomeTeam": "E", "AwayTeam": "F", "FTR": "D", "Season": "2526"},
    ])
    errors, _ = validate_clean_history(frame, ["2627", "2526"], min_season_rows=2)
    assert errors == []

    truncated = frame[frame["Season"] == "2627"]
    errors, _ = validate_clean_history(truncated, ["2627", "2526"], min_season_rows=2)
    assert any("2526" in message for message in errors)
