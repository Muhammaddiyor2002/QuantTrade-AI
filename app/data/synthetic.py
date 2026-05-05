"""Deterministic synthetic OHLCV generator (offline / test-friendly)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.utils.timeframes import timeframe_to_minutes


def generate_ohlcv(
    n: int = 1_000,
    *,
    timeframe: str = "1h",
    start: str | pd.Timestamp = "2023-01-01",
    seed: int = 42,
    initial_price: float = 100.0,
    drift: float = 0.0001,
    volatility: float = 0.01,
) -> pd.DataFrame:
    """Generate a deterministic geometric Brownian motion OHLCV frame.

    Returns a DataFrame indexed by `timestamp` with columns
    ``[open, high, low, close, volume]``.
    """
    if n <= 1:
        raise ValueError("n must be > 1")
    rng = np.random.default_rng(seed)
    minutes = timeframe_to_minutes(timeframe)
    idx = pd.date_range(start=pd.Timestamp(start), periods=n, freq=f"{minutes}min")

    log_returns = rng.normal(loc=drift, scale=volatility, size=n)
    close = initial_price * np.exp(np.cumsum(log_returns))

    open_ = np.empty_like(close)
    open_[0] = initial_price
    open_[1:] = close[:-1]

    intra = np.abs(rng.normal(0.0, volatility, size=(n, 2)))
    high = np.maximum(open_, close) * (1.0 + intra[:, 0])
    low = np.minimum(open_, close) * (1.0 - intra[:, 1])
    volume = rng.lognormal(mean=10.0, sigma=0.4, size=n)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df
