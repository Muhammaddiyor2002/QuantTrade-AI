from __future__ import annotations

import pandas as pd
import pytest

from app.strategies import (
    BreakoutStrategy,
    MovingAverageCrossover,
    RSIStrategy,
    crossover,
    crossunder,
    get_strategy,
    list_strategies,
)


def test_registry_lists() -> None:
    names = list_strategies()
    assert "ma_crossover" in names
    assert "rsi" in names
    assert "breakout" in names


def test_registry_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_strategy("does_not_exist")


def test_ma_crossover_signal_values(ohlcv_medium: pd.DataFrame) -> None:
    strat = MovingAverageCrossover(fast=10, slow=30)
    sigs = strat.generate_signals(ohlcv_medium)
    assert sigs.position.isin([-1, 0, 1]).all()
    assert len(sigs.position) == len(ohlcv_medium)


def test_ma_crossover_invalid_periods() -> None:
    with pytest.raises(ValueError):
        MovingAverageCrossover(fast=50, slow=20)


def test_rsi_strategy_long_only(ohlcv_medium: pd.DataFrame) -> None:
    strat = RSIStrategy()
    sigs = strat.generate_signals(ohlcv_medium)
    assert sigs.position.between(0, 1).all()


def test_rsi_strategy_short(ohlcv_medium: pd.DataFrame) -> None:
    strat = RSIStrategy(allow_short=True)
    sigs = strat.generate_signals(ohlcv_medium)
    assert sigs.position.between(-1, 1).all()


def test_breakout_strategy(ohlcv_medium: pd.DataFrame) -> None:
    strat = BreakoutStrategy(entry_window=20, exit_window=10)
    sigs = strat.generate_signals(ohlcv_medium)
    assert sigs.position.isin([-1, 0, 1]).all()


def test_crossover_helpers() -> None:
    a = pd.Series([1, 2, 3, 4, 3])
    b = pd.Series([2, 2, 2, 2, 5])
    co = crossover(a, b)
    cu = crossunder(a, b)
    assert co.iloc[2] == True  # noqa: E712
    assert cu.iloc[4] == True  # noqa: E712
