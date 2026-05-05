"""Strategy base class and shared types.

Strategies are *vectorized*: given an OHLCV DataFrame they return a
``Signals`` object describing the desired position at each bar. The
backtesting engine consumes that to simulate trades.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class Signals:
    """Output of a strategy.

    ``position`` is in {-1, 0, 1} (short/flat/long). The backtester
    interprets a change in position as an entry/exit/reverse.
    """

    position: pd.Series
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.position, pd.Series):
            raise TypeError("position must be a pandas Series")
        if not self.position.isin([-1, 0, 1]).all():
            raise ValueError("position values must be in {-1, 0, 1}")


class Strategy(ABC):
    """Abstract base class for vectorized strategies."""

    name: str = "strategy"

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = params

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> Signals:
        """Return :class:`Signals` for `df`."""

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({param_str})"


def positions_from_long_signal(long_signal: pd.Series) -> pd.Series:
    """Convert a boolean long-only signal into {0,1} position with prior fill."""
    pos = long_signal.astype("int64")
    return pos.fillna(0).astype("int64")


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """Return boolean Series where `a` crosses above `b`."""
    diff = (a - b).astype(float)
    sign = np.sign(diff)
    return (sign > 0) & (sign.shift(1) <= 0)


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    """Return boolean Series where `a` crosses below `b`."""
    diff = (a - b).astype(float)
    sign = np.sign(diff)
    return (sign < 0) & (sign.shift(1) >= 0)
