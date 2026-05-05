"""Trend indicators: SMA, EMA, MACD."""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, period: int = 20) -> pd.Series:
    """Simple moving average."""
    if period <= 0:
        raise ValueError("period must be positive")
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int = 20) -> pd.Series:
    """Exponential moving average (adjust=False, RMA-equivalent smoothing)."""
    if period <= 0:
        raise ValueError("period must be positive")
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Return DataFrame with columns ``macd``, ``signal``, ``hist``."""
    if fast >= slow:
        raise ValueError("fast must be less than slow")
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "hist": macd_line - signal_line,
        }
    )
