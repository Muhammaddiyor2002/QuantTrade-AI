"""Volatility indicators: Bollinger Bands, ATR."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands (mid, upper, lower, bandwidth, %b)."""
    if period <= 0:
        raise ValueError("period must be positive")
    mid = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (series - lower) / (upper - lower).replace(0.0, np.nan)
    bandwidth = (upper - lower) / mid.replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_pct_b": pct_b,
            "bb_bandwidth": bandwidth,
        }
    )


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    if period <= 0:
        raise ValueError("period must be positive")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().rename("atr")
