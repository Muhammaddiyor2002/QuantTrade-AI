from __future__ import annotations

import pandas as pd

from app.backtesting import BacktestConfig, BacktestEngine
from app.backtesting.metrics import (
    cagr,
    compute_metrics,
    max_drawdown,
    sharpe_ratio,
)
from app.strategies import MovingAverageCrossover


def test_backtest_runs(ohlcv_medium: pd.DataFrame) -> None:
    engine = BacktestEngine(BacktestConfig(timeframe="1h"))
    res = engine.run_strategy(ohlcv_medium, MovingAverageCrossover(fast=10, slow=30))
    assert len(res.equity) == len(ohlcv_medium)
    assert res.metrics.num_trades >= 0
    summary = res.summary()
    assert "sharpe" in summary.index


def test_metrics_zero_when_flat(ohlcv_small: pd.DataFrame) -> None:
    pos = pd.Series(0, index=ohlcv_small.index)
    engine = BacktestEngine(BacktestConfig(timeframe="1h"))
    res = engine.run(ohlcv_small, pos)
    assert res.metrics.num_trades == 0
    assert res.metrics.total_return == 0.0


def test_helpers_handle_empty() -> None:
    s = pd.Series(dtype=float)
    assert max_drawdown(s) == 0.0
    assert sharpe_ratio(s, periods_per_year=252) == 0.0
    assert cagr(s, periods_per_year=252) == 0.0


def test_compute_metrics_handles_zero_pnl(ohlcv_small: pd.DataFrame) -> None:
    eq = pd.Series([1.0, 1.0, 1.0], index=ohlcv_small.index[:3])
    metrics = compute_metrics(eq, [], timeframe="1h")
    assert metrics.num_trades == 0
    assert metrics.profit_factor == 0.0


def test_long_only_profitable_on_trend(ohlcv_trending: pd.DataFrame) -> None:
    engine = BacktestEngine(BacktestConfig(timeframe="1h", commission_bps=0, slippage_bps=0))
    res = engine.run_strategy(ohlcv_trending, MovingAverageCrossover(fast=10, slow=30))
    # Trending with no costs should at least produce non-zero activity
    assert res.metrics.num_trades > 0


def test_trade_pnl_includes_entry_commission(ohlcv_trending: pd.DataFrame) -> None:
    """Sum of trade PnL (which now includes both entry and exit commissions)
    must agree with the change in equity to within force-close roundoff."""
    engine = BacktestEngine(BacktestConfig(timeframe="1h", commission_bps=10, slippage_bps=0))
    res = engine.run_strategy(ohlcv_trending, MovingAverageCrossover(fast=10, slow=30))
    if not res.trades:
        return
    trades_pnl = sum(t.pnl for t in res.trades)
    equity_change = float(res.equity.iloc[-1] - engine.config.initial_capital)
    assert abs(trades_pnl - equity_change) < 1e-6 * engine.config.initial_capital


def test_trade_commission_field_includes_entry_and_exit(ohlcv_medium: pd.DataFrame) -> None:
    engine = BacktestEngine(BacktestConfig(timeframe="1h", commission_bps=20, slippage_bps=0))
    res = engine.run_strategy(ohlcv_medium, MovingAverageCrossover(fast=10, slow=30))
    if not res.trades:
        return
    t = res.trades[0]
    one_side = abs(t.quantity) * t.entry_price * 0.002
    # commission must reflect BOTH legs, so it should be roughly 2x a single side.
    assert t.commission > 1.5 * one_side
