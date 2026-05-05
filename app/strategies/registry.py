"""Strategy registry — discover/instantiate strategies by name."""

from __future__ import annotations

from typing import Any

from app.strategies.base import Strategy
from app.strategies.breakout import BreakoutStrategy
from app.strategies.ma_crossover import MovingAverageCrossover
from app.strategies.rsi import RSIStrategy

_REGISTRY: dict[str, type[Strategy]] = {
    MovingAverageCrossover.name: MovingAverageCrossover,
    RSIStrategy.name: RSIStrategy,
    BreakoutStrategy.name: BreakoutStrategy,
}


def list_strategies() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_strategy(name: str, **params: Any) -> Strategy:
    """Instantiate a strategy by registered name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown strategy: {name!r}. Available: {list_strategies()}")
    return _REGISTRY[name](**params)


def register_strategy(cls: type[Strategy]) -> type[Strategy]:
    """Decorator-style registration for custom strategies."""
    _REGISTRY[cls.name] = cls
    return cls
