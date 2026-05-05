"""Backtesting engine and performance metrics."""

from app.backtesting.engine import BacktestConfig, BacktestEngine, BacktestResult
from app.backtesting.metrics import (
    PerformanceMetrics,
    cagr,
    compute_metrics,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from app.backtesting.trade import Trade

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "PerformanceMetrics",
    "Trade",
    "cagr",
    "compute_metrics",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
]
