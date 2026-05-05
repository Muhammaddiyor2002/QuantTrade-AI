"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from app.data.synthetic import generate_ohlcv


@pytest.fixture(scope="session")
def ohlcv_small() -> pd.DataFrame:
    return generate_ohlcv(n=300, timeframe="1h", seed=1, initial_price=100.0)


@pytest.fixture(scope="session")
def ohlcv_medium() -> pd.DataFrame:
    return generate_ohlcv(n=1_000, timeframe="1h", seed=7, initial_price=100.0)


@pytest.fixture(scope="session")
def ohlcv_trending() -> pd.DataFrame:
    """A series with positive drift so MA-crossover should be profitable."""
    return generate_ohlcv(
        n=1_500,
        timeframe="1h",
        seed=99,
        initial_price=50.0,
        drift=0.0004,
        volatility=0.005,
    )
