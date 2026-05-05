from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.indicators import attach_default_indicators
from app.indicators.momentum import rsi, stochastic
from app.indicators.trend import ema, macd, sma
from app.indicators.volatility import atr, bollinger_bands
from app.indicators.volume import obv, vwap


def _ramp(n: int = 100) -> pd.Series:
    return pd.Series(np.linspace(1.0, 100.0, n), name="x")


def test_sma_basic() -> None:
    s = _ramp(20)
    result = sma(s, period=5)
    assert result.iloc[:4].isna().all()
    assert pytest.approx(result.iloc[-1]) == s.iloc[-5:].mean()


def test_ema_smoothing() -> None:
    s = _ramp(50)
    result = ema(s, period=10)
    assert result.iloc[-1] > result.iloc[-20]


def test_macd_columns() -> None:
    s = _ramp(200)
    df = macd(s)
    assert set(df.columns) == {"macd", "signal", "hist"}


def test_rsi_bounds(ohlcv_small: pd.DataFrame) -> None:
    rsi_v = rsi(ohlcv_small["close"], period=14).dropna()
    assert ((rsi_v >= 0) & (rsi_v <= 100)).all()


def test_bollinger_consistency(ohlcv_small: pd.DataFrame) -> None:
    bb = bollinger_bands(ohlcv_small["close"]).dropna()
    assert (bb["bb_upper"] >= bb["bb_mid"]).all()
    assert (bb["bb_mid"] >= bb["bb_lower"]).all()


def test_atr_positive(ohlcv_small: pd.DataFrame) -> None:
    a = atr(ohlcv_small["high"], ohlcv_small["low"], ohlcv_small["close"]).dropna()
    assert (a > 0).all()


def test_volume_indicators(ohlcv_small: pd.DataFrame) -> None:
    o = obv(ohlcv_small["close"], ohlcv_small["volume"])
    v = vwap(ohlcv_small["high"], ohlcv_small["low"], ohlcv_small["close"], ohlcv_small["volume"])
    assert len(o) == len(ohlcv_small)
    assert v.dropna().notna().all()


def test_stochastic_bounds(ohlcv_small: pd.DataFrame) -> None:
    s = stochastic(ohlcv_small["high"], ohlcv_small["low"], ohlcv_small["close"]).dropna()
    assert s["stoch_k"].between(0, 100).all()


def test_attach_default_indicators(ohlcv_small: pd.DataFrame) -> None:
    out = attach_default_indicators(ohlcv_small)
    for col in ["sma_20", "ema_20", "rsi_14", "macd", "bb_upper", "atr_14", "obv"]:
        assert col in out.columns
