"""Performance metrics for an equity curve / trade list."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.backtesting.trade import Trade
from app.utils.timeframes import bars_per_year


@dataclass(slots=True)
class PerformanceMetrics:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    expectancy: float
    num_trades: int
    exposure: float  # fraction of bars in market

    def to_dict(self) -> dict[str, float]:
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_pnl": self.avg_trade_pnl,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "expectancy": self.expectancy,
            "num_trades": float(self.num_trades),
            "exposure": self.exposure,
        }


def equity_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().fillna(0.0)


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min()) if len(dd) else 0.0


def sharpe_ratio(returns: pd.Series, periods_per_year: float, rf: float = 0.0) -> float:
    excess = returns - rf / periods_per_year
    sd = excess.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: float, rf: float = 0.0) -> float:
    excess = returns - rf / periods_per_year
    downside = excess.clip(upper=0.0)
    dd_std = np.sqrt(np.mean(downside**2))
    if dd_std == 0 or np.isnan(dd_std):
        return 0.0
    return float(excess.mean() / dd_std * np.sqrt(periods_per_year))


def cagr(equity: pd.Series, periods_per_year: float) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


def compute_metrics(
    equity: pd.Series,
    trades: list[Trade],
    *,
    timeframe: str = "1h",
    in_market: pd.Series | None = None,
    annualization_mode: str = "crypto",
) -> PerformanceMetrics:
    """Compute the standard performance metrics for an equity curve and trade list."""
    if equity.empty:
        return PerformanceMetrics(
            total_return=0.0,
            cagr=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown=0.0,
            calmar=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_trade_pnl=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            expectancy=0.0,
            num_trades=0,
            exposure=0.0,
        )

    ppy = bars_per_year(timeframe, mode=annualization_mode)
    rets = equity_returns(equity)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    mdd = max_drawdown(equity)
    sharpe = sharpe_ratio(rets, ppy)
    sortino = sortino_ratio(rets, ppy)
    cagr_val = cagr(equity, ppy)
    calmar = (cagr_val / abs(mdd)) if mdd != 0 else 0.0

    pnls = np.array([t.pnl for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    num = len(pnls)
    win_rate = float(len(wins) / num) if num else 0.0
    profit_factor = (
        float(wins.sum() / -losses.sum())
        if losses.size and losses.sum() != 0
        else (float("inf") if wins.size else 0.0)
    )
    avg_pnl = float(pnls.mean()) if num else 0.0
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0
    expectancy = float(win_rate * avg_win + (1.0 - win_rate) * avg_loss) if num else 0.0

    if in_market is not None and len(in_market):
        exposure = float((in_market.astype(bool)).mean())
    else:
        exposure = 0.0

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr_val,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        calmar=float(calmar),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_pnl=avg_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy=expectancy,
        num_trades=num,
        exposure=exposure,
    )
