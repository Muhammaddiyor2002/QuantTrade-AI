"""Feature engineering for ML price-direction models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.indicators import attach_default_indicators


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a feature matrix from OHLCV input.

    The frame is enriched with default indicators and a panel of
    log-return / volatility / volume features over multiple horizons.
    """
    out = attach_default_indicators(df)
    close = out["close"]
    out["ret_1"] = np.log(close).diff()
    for h in (3, 5, 10, 20):
        out[f"ret_{h}"] = np.log(close).diff(h)
        out[f"vol_{h}"] = out["ret_1"].rolling(h, min_periods=h).std()
    out["range_pct"] = (out["high"] - out["low"]) / out["close"].replace(0.0, np.nan)
    out["body_pct"] = (out["close"] - out["open"]) / out["close"].replace(0.0, np.nan)
    out["vol_ma_20"] = out["volume"].rolling(20, min_periods=20).mean()
    out["vol_ratio"] = out["volume"] / out["vol_ma_20"].replace(0.0, np.nan)
    return out


def make_target(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Binary target: 1 if forward `horizon`-bar log-return is positive."""
    fwd = np.log(close).diff(horizon).shift(-horizon)
    return (fwd > 0).astype("int64").rename("target")


FEATURE_COLUMNS: list[str] = [
    "sma_20",
    "sma_50",
    "ema_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_lower",
    "bb_mid",
    "atr_14",
    "obv",
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "vol_3",
    "vol_5",
    "vol_10",
    "vol_20",
    "range_pct",
    "body_pct",
    "vol_ratio",
]


def build_xy(
    df: pd.DataFrame,
    *,
    horizon: int = 1,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return aligned ``(X, y)`` ready for an ML model."""
    feats = make_features(df)
    target = make_target(feats["close"], horizon=horizon)
    cols = feature_columns or FEATURE_COLUMNS
    X = feats[cols].copy()
    y = target
    aligned = pd.concat([X, y], axis=1).dropna()
    return aligned[cols], aligned["target"].astype(int)
