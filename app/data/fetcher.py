"""Market data fetcher.

Tries `ccxt` first (live exchange data); falls back to a synthetic generator if
the exchange is unreachable or `ccxt` is unavailable. This keeps the platform
fully usable offline (CI, tests, demos) while still supporting live data when
network access is present.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from app.data.synthetic import generate_ohlcv
from app.utils.logger import get_logger
from app.utils.timeframes import timeframe_to_minutes

log = get_logger(__name__)


@dataclass(slots=True)
class OHLCVRequest:
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    limit: int = 1_000
    since: int | None = None  # ms since epoch
    exchange: str = "binance"


def _fetch_ccxt(req: OHLCVRequest) -> pd.DataFrame:
    """Fetch via ccxt. Raises on any error."""
    import ccxt  # type: ignore[import-untyped]

    cls = getattr(ccxt, req.exchange, None)
    if cls is None:
        raise ValueError(f"Unknown exchange: {req.exchange!r}")
    client = cls({"enableRateLimit": True})
    raw = client.fetch_ohlcv(req.symbol, timeframe=req.timeframe, since=req.since, limit=req.limit)
    if not raw:
        raise RuntimeError("Empty OHLCV response")
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    return df


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 1_000,
    *,
    exchange: str = "binance",
    fallback_to_synthetic: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV data. Falls back to deterministic synthetic data on failure.

    The returned DataFrame has columns ``[open, high, low, close, volume]`` and a
    ``DatetimeIndex`` named ``timestamp`` in ascending order.
    """
    req = OHLCVRequest(symbol=symbol, timeframe=timeframe, limit=limit, exchange=exchange)
    try:
        df = _fetch_ccxt(req)
        log.info("Fetched %d bars for %s %s from %s", len(df), symbol, timeframe, exchange)
        return df
    except Exception as exc:
        if not fallback_to_synthetic:
            raise
        log.warning(
            "ccxt fetch failed (%s). Falling back to synthetic data for %s %s.",
            exc,
            symbol,
            timeframe,
        )
        seed = abs(hash((symbol, timeframe, exchange))) % (2**32)
        df = generate_ohlcv(n=limit, timeframe=timeframe, seed=seed)
        return df


def now_ms() -> int:
    return int(datetime.now(tz=UTC).timestamp() * 1000)


def since_n_bars(n: int, timeframe: str) -> int:
    """Return a ms-epoch ``since`` covering the last *n* bars of `timeframe`."""
    minutes = timeframe_to_minutes(timeframe)
    return now_ms() - n * minutes * 60 * 1000
