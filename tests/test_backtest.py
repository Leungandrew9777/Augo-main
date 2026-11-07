import numpy as np
import pandas as pd

from backtest import _bootstrap_mean_ci, season_start_year, season_test_folds


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
