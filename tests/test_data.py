from __future__ import annotations

import pandas as pd

from app.data.fetcher import fetch_ohlcv, since_n_bars
from app.data.storage import load_ohlcv, save_ohlcv
from app.data.synthetic import generate_ohlcv


def test_synthetic_shape_and_columns() -> None:
    df = generate_ohlcv(n=500, timeframe="1h", seed=3)
    assert len(df) == 500
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing
    assert (df["high"] >= df["low"]).all()
    assert (df["volume"] > 0).all()


def test_synthetic_deterministic() -> None:
    a = generate_ohlcv(n=200, seed=42)
    b = generate_ohlcv(n=200, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_fetch_falls_back_to_synthetic() -> None:
    df = fetch_ohlcv(symbol="ZZZZ/USDT", timeframe="1h", limit=200, exchange="binance")
    assert not df.empty
    assert len(df) == 200


def test_storage_roundtrip(tmp_path) -> None:
    df = generate_ohlcv(n=200, seed=5)
    save_ohlcv(df, "BTC/USDT", "1h", base=tmp_path)
    df2 = load_ohlcv("BTC/USDT", "1h", base=tmp_path)
    assert len(df2) == len(df)


def test_since_n_bars_positive() -> None:
    assert since_n_bars(10, "1h") > 0
