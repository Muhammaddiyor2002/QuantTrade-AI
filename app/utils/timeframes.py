"""Helpers for parsing/normalizing timeframes."""

from __future__ import annotations

# Number of bars per year for common timeframes (assuming crypto 24/7).
_BARS_PER_YEAR_CRYPTO: dict[str, float] = {
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "30m": 17_520,
    "1h": 8_760,
    "2h": 4_380,
    "4h": 2_190,
    "6h": 1_460,
    "12h": 730,
    "1d": 365,
    "1w": 52,
}

# Equity-style annualization (252 trading days).
_BARS_PER_YEAR_EQUITY: dict[str, float] = {
    "1m": 252 * 6.5 * 60,
    "5m": 252 * 6.5 * 12,
    "15m": 252 * 6.5 * 4,
    "30m": 252 * 6.5 * 2,
    "1h": 252 * 6.5,
    "1d": 252,
    "1w": 52,
}


def bars_per_year(timeframe: str, *, mode: str = "crypto") -> float:
    """Return number of bars/year for `timeframe` ("crypto" 24/7 or "equity" 252)."""
    table = _BARS_PER_YEAR_EQUITY if mode == "equity" else _BARS_PER_YEAR_CRYPTO
    if timeframe not in table:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    return table[timeframe]


def timeframe_to_minutes(timeframe: str) -> int:
    """Convert a textual timeframe into minutes."""
    units = {"m": 1, "h": 60, "d": 60 * 24, "w": 60 * 24 * 7}
    if not timeframe:
        raise ValueError("Empty timeframe")
    suffix = timeframe[-1]
    if suffix not in units:
        raise ValueError(f"Unsupported timeframe suffix: {suffix!r}")
    try:
        n = int(timeframe[:-1])
    except ValueError as e:
        raise ValueError(f"Invalid timeframe: {timeframe!r}") from e
    return n * units[suffix]
