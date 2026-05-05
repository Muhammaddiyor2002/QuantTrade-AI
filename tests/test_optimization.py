from __future__ import annotations

import pandas as pd

from app.backtesting import BacktestConfig
from app.optimization import compare_strategies, grid_search, walk_forward


def test_grid_search_runs(ohlcv_medium: pd.DataFrame) -> None:
    res = grid_search(
        ohlcv_medium,
        "ma_crossover",
        {"fast": [10, 20], "slow": [50, 80]},
        config=BacktestConfig(timeframe="1h"),
    )
    assert len(res.leaderboard) == 4
    assert "fast" in res.best_params
    assert "slow" in res.best_params


def test_compare_strategies(ohlcv_medium: pd.DataFrame) -> None:
    df = compare_strategies(
        ohlcv_medium,
        [
            ("ma_crossover", {"fast": 10, "slow": 30}),
            ("rsi", {"period": 14}),
        ],
        config=BacktestConfig(timeframe="1h"),
    )
    assert len(df) == 2
    assert "sharpe" in df.columns


def test_walk_forward(ohlcv_trending: pd.DataFrame) -> None:
    rep = walk_forward(
        ohlcv_trending,
        "ma_crossover",
        {"fast": [10, 20], "slow": [50, 100]},
        n_folds=3,
        config=BacktestConfig(timeframe="1h"),
    )
    assert len(rep.folds) == 3
    assert "oos_sharpe" in rep.summary.columns
    assert not rep.aggregated_equity.empty
