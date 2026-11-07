import numpy as np
import pandas as pd

from backtest import _bootstrap_mean_ci, feature_columns, season_start_year, season_test_folds
from feature_engineering import EURO_ELO_COLS, EURO_REST_COLS
from features import MODEL_FEATURE_COLS


def test_season_start_year():
    assert season_start_year("1314") == 2013
    assert season_start_year("2021") == 2020
    assert season_start_year("2627") == 2026


def test_season_folds_use_only_earlier_seasons():
    frame = pd.DataFrame({
        "Season": ["2425"] * 250 + ["2526"] * 250 + ["2627"] * 40,
        "Date": pd.date_range("2024-08-01", periods=540, freq="D"),
        "Result": [0] * 540,
    })
    folds = season_test_folds(frame, n_seasons=1)
    assert [fold[0] for fold in folds] == ["2526", "2627"]

    _season, train, test = folds[0]
    assert len(test) == 250
    assert set(train["Season"].unique()) == {"2425"}

    _season, train, test = folds[1]
    assert set(train["Season"].unique()) == {"2425", "2526"}


def test_bootstrap_ci_contains_mean():
    values = np.full(100, 0.25)
    low, high = _bootstrap_mean_ci(values)
    assert low == 0.25
    assert high == 0.25


def test_feature_columns_groups():
    columns = MODEL_FEATURE_COLS + EURO_ELO_COLS + EURO_REST_COLS
    frame = pd.DataFrame(columns=columns)

    baseline = feature_columns("baseline", frame)
    assert baseline == MODEL_FEATURE_COLS

    euroelo = feature_columns("euroelo", frame)
    assert set(euroelo) == set(MODEL_FEATURE_COLS) | set(EURO_ELO_COLS)

    rest = feature_columns("eurorest", frame)
    assert set(rest) == set(MODEL_FEATURE_COLS) | set(EURO_REST_COLS)

    combined = feature_columns("all", frame)
    assert set(combined) == set(columns)
