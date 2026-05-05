"""Vectorized event-aware backtesting engine.

The engine consumes a price frame and a position series ({-1, 0, 1}). It
accounts for commission (bps) and slippage (bps) on every fill, and sizes
positions either as a constant fraction of equity (``fraction``) or via a
risk-based sizer fed by ATR (``risk_per_trade``).

Key outputs (:class:`BacktestResult`):

* ``equity`` - portfolio equity curve indexed by bar.
* ``returns`` - per-bar simple returns.
* ``positions`` - integer position per bar.
* ``trades`` - list of closed :class:`Trade` instances.
* ``metrics`` - :class:`PerformanceMetrics`.

The implementation is intentionally simple and well-tested rather than
hyper-realistic: signals act on the *next* bar's open price (no look-ahead),
and the backtest is a single-asset cash + position simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.backtesting.metrics import PerformanceMetrics, compute_metrics
from app.backtesting.trade import Trade
from app.strategies.base import Signals, Strategy
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    commission_bps: float = 5.0
    slippage_bps: float = 2.0
    fraction: float = 1.0  # fraction of equity to deploy per trade
    timeframe: str = "1h"
    annualization_mode: str = "crypto"


@dataclass(slots=True)
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: list[Trade]
    metrics: PerformanceMetrics
    config: BacktestConfig
    extras: dict = field(default_factory=dict)

    def summary(self) -> pd.Series:
        return pd.Series(self.metrics.to_dict())


def _apply_costs(price: float, side: int, commission_bps: float, slippage_bps: float) -> float:
    """Return the effective fill price after slippage. `side` is +1 buy, -1 sell."""
    slippage = price * slippage_bps / 10_000.0 * side
    return price + slippage


class BacktestEngine:
    """Vectorized backtester that turns a position series into an equity curve."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(
        self,
        df: pd.DataFrame,
        signals: Signals | pd.Series,
    ) -> BacktestResult:
        if not {"open", "high", "low", "close"}.issubset(df.columns):
            raise ValueError("df must contain OHLC columns")
        positions = signals.position if isinstance(signals, Signals) else signals
        if not isinstance(positions, pd.Series):
            raise TypeError("positions must be a pandas Series")

        positions = positions.reindex(df.index).fillna(0).astype(int)
        # Avoid look-ahead: the position decided at bar t is filled at bar t+1's open.
        target = positions.shift(1).fillna(0).astype(int)

        n = len(df)
        equity = np.empty(n, dtype=float)
        cash = self.config.initial_capital
        qty = 0.0
        entry_price = 0.0
        entry_time: pd.Timestamp | None = None
        entry_idx = 0
        entry_commission = 0.0
        cur_dir = 0
        trades: list[Trade] = []
        commission_rate = self.config.commission_bps / 10_000.0
        opens = df["open"].to_numpy()
        closes = df["close"].to_numpy()

        for i in range(n):
            desired = int(target.iloc[i])
            price = opens[i]

            if desired != cur_dir:
                # Close existing position
                if cur_dir != 0 and qty != 0.0:
                    fill = _apply_costs(
                        price, -cur_dir, commission_rate * 10_000, self.config.slippage_bps
                    )
                    exit_commission = abs(qty) * fill * commission_rate
                    cash += qty * fill - exit_commission
                    gross_pnl = (fill - entry_price) * qty
                    total_commission = entry_commission + exit_commission
                    trade = Trade(
                        entry_time=entry_time or df.index[entry_idx],
                        exit_time=df.index[i],
                        direction=cur_dir,
                        entry_price=entry_price,
                        exit_price=fill,
                        quantity=qty,
                        pnl=float(gross_pnl - total_commission),
                        pnl_pct=float((fill / entry_price - 1.0) * cur_dir) if entry_price else 0.0,
                        commission=float(total_commission),
                        slippage=0.0,
                        bars_held=int(i - entry_idx),
                    )
                    trades.append(trade)
                    qty = 0.0
                    cur_dir = 0
                    entry_commission = 0.0

                # Open new position
                if desired != 0:
                    fill = _apply_costs(
                        price, desired, commission_rate * 10_000, self.config.slippage_bps
                    )
                    notional = cash * self.config.fraction
                    qty = (notional / fill) * desired if fill > 0 else 0.0
                    entry_commission = abs(qty) * fill * commission_rate
                    cash -= qty * fill + entry_commission
                    entry_price = fill
                    entry_time = df.index[i]
                    entry_idx = i
                    cur_dir = desired

            mark = closes[i]
            equity[i] = cash + qty * mark

        # Force-close any open position at the last bar's close
        if cur_dir != 0 and qty != 0.0:
            fill = _apply_costs(closes[-1], -cur_dir, 0.0, self.config.slippage_bps)
            exit_commission = abs(qty) * fill * commission_rate
            cash += qty * fill - exit_commission
            gross_pnl = (fill - entry_price) * qty
            total_commission = entry_commission + exit_commission
            trades.append(
                Trade(
                    entry_time=entry_time or df.index[entry_idx],
                    exit_time=df.index[-1],
                    direction=cur_dir,
                    entry_price=entry_price,
                    exit_price=fill,
                    quantity=qty,
                    pnl=float(gross_pnl - total_commission),
                    pnl_pct=float((fill / entry_price - 1.0) * cur_dir) if entry_price else 0.0,
                    commission=float(total_commission),
                    slippage=0.0,
                    bars_held=int(n - 1 - entry_idx),
                )
            )
            equity[-1] = cash

        eq = pd.Series(equity, index=df.index, name="equity")
        rets = eq.pct_change().fillna(0.0)
        in_market = (target != 0).astype(int)

        metrics = compute_metrics(
            eq,
            trades,
            timeframe=self.config.timeframe,
            in_market=in_market,
            annualization_mode=self.config.annualization_mode,
        )
        log.info(
            "Backtest done: %d trades | total_return=%.2f%% | sharpe=%.2f | mdd=%.2f%%",
            len(trades),
            metrics.total_return * 100,
            metrics.sharpe,
            metrics.max_drawdown * 100,
        )
        return BacktestResult(
            equity=eq,
            returns=rets,
            positions=target,
            trades=trades,
            metrics=metrics,
            config=self.config,
        )

    def run_strategy(self, df: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        signals = strategy.generate_signals(df)
        return self.run(df, signals)
