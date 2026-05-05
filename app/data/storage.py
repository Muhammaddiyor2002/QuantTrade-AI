"""Local OHLCV storage backed by Parquet files (per symbol+timeframe)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)


def _slug(symbol: str, timeframe: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", symbol).strip("_")
    return f"{s}_{timeframe}"


def _path(symbol: str, timeframe: str, base: Path | None = None) -> Path:
    base_dir = base or get_settings().data_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{_slug(symbol, timeframe)}.parquet"


def save_ohlcv(df: pd.DataFrame, symbol: str, timeframe: str, base: Path | None = None) -> Path:
    """Persist OHLCV to Parquet, deduplicating by index against any existing file."""
    p = _path(symbol, timeframe, base)
    if p.exists():
        existing = pd.read_parquet(p)
        df = pd.concat([existing, df])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(p)
    log.info("Saved %d rows -> %s", len(df), p)
    return p


def load_ohlcv(symbol: str, timeframe: str, base: Path | None = None) -> pd.DataFrame:
    """Load OHLCV from Parquet. Returns empty DataFrame if not present."""
    p = _path(symbol, timeframe, base)
    if not p.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.read_parquet(p)
