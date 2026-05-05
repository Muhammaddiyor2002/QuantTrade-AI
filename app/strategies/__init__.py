"""Strategy library."""

from app.strategies.base import Signals, Strategy, crossover, crossunder
from app.strategies.breakout import BreakoutStrategy
from app.strategies.ma_crossover import MovingAverageCrossover
from app.strategies.registry import get_strategy, list_strategies, register_strategy
from app.strategies.rsi import RSIStrategy

__all__ = [
    "BreakoutStrategy",
    "MovingAverageCrossover",
    "RSIStrategy",
    "Signals",
    "Strategy",
    "crossover",
    "crossunder",
    "get_strategy",
    "list_strategies",
    "register_strategy",
]
