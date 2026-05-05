"""Momentum indicators: RSI, Stochastic."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI on the input close series.

    Uses Wilder's smoothing (RMA) which matches the original definition.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.where(avg_loss != 0.0, 100.0)
    return out.rename("rsi")


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """Classic %K / %D stochastic oscillator."""
    lowest = low.rolling(k_period, min_periods=k_period).min()
    highest = high.rolling(k_period, min_periods=k_period).max()
    denom = (highest - lowest).replace(0.0, np.nan)
    k = 100.0 * (close - lowest) / denom
    d = k.rolling(d_period, min_periods=d_period).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})
