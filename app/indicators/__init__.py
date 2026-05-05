"""Vectorized technical indicators."""

from __future__ import annotations

import pandas as pd

from app.indicators.momentum import rsi, stochastic
from app.indicators.trend import ema, macd, sma
from app.indicators.volatility import atr, bollinger_bands
from app.indicators.volume import obv, vwap

__all__ = [
    "atr",
    "attach_default_indicators",
    "bollinger_bands",
    "ema",
    "macd",
    "obv",
    "rsi",
    "sma",
    "stochastic",
    "vwap",
]


def attach_default_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` enriched with a standard indicator panel."""
    if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
        raise ValueError("DataFrame must contain OHLCV columns")
    out = df.copy()
    out["sma_20"] = sma(out["close"], 20)
    out["sma_50"] = sma(out["close"], 50)
    out["ema_20"] = ema(out["close"], 20)
    out["rsi_14"] = rsi(out["close"], 14)
    macd_df = macd(out["close"])
    out[["macd", "macd_signal", "macd_hist"]] = macd_df.values
    bb = bollinger_bands(out["close"])
    out["bb_upper"] = bb["bb_upper"]
    out["bb_lower"] = bb["bb_lower"]
    out["bb_mid"] = bb["bb_mid"]
    out["atr_14"] = atr(out["high"], out["low"], out["close"], 14)
    out["obv"] = obv(out["close"], out["volume"])
    return out
