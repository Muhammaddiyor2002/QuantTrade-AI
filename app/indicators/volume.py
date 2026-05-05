"""Volume indicators: OBV, VWAP."""

from __future__ import annotations

import numpy as np
import pandas as pd


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum().rename("obv")


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume-Weighted Average Price (cumulative)."""
    typical = (high + low + close) / 3.0
    cum_pv = (typical * volume).cumsum()
    cum_v = volume.cumsum().replace(0.0, np.nan)
    return (cum_pv / cum_v).rename("vwap")
