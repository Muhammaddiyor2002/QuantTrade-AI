"""Compare several strategies on the same dataset."""

from __future__ import annotations

import pandas as pd

from app.backtesting import BacktestConfig, BacktestEngine
from app.strategies import get_strategy


def compare_strategies(
    df: pd.DataFrame,
    strategies: list[tuple[str, dict]],
    *,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Run each (strategy_name, params) pair and return a metrics leaderboard."""
    engine = BacktestEngine(config or BacktestConfig())
    rows = []
    for name, params in strategies:
        strat = get_strategy(name, **params)
        result = engine.run_strategy(df, strat)
        rows.append({"strategy": name, **params, **result.metrics.to_dict()})
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
