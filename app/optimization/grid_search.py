"""Exhaustive grid-search hyperparameter optimization for rule-based strategies."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import pandas as pd

from app.backtesting import BacktestConfig, BacktestEngine
from app.strategies import get_strategy
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class GridSearchResult:
    best_params: dict[str, Any]
    best_score: float
    leaderboard: pd.DataFrame


def grid_search(
    df: pd.DataFrame,
    strategy_name: str,
    grid: dict[str, list[Any]],
    *,
    config: BacktestConfig | None = None,
    metric: str = "sharpe",
) -> GridSearchResult:
    """Brute-force scan a parameter grid; rank by `metric`."""
    if not grid:
        raise ValueError("grid must not be empty")
    engine = BacktestEngine(config or BacktestConfig())
    keys = list(grid.keys())
    rows: list[dict[str, Any]] = []
    param_sets: list[dict[str, Any]] = []
    for combo in product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo, strict=False))
        try:
            strat = get_strategy(strategy_name, **params)
        except Exception as exc:
            log.debug("Skip invalid combo %s (%s)", params, exc)
            continue
        result = engine.run_strategy(df, strat)
        rows.append({**params, **result.metrics.to_dict()})
        param_sets.append(params)
    if not rows:
        raise RuntimeError("No valid parameter combinations evaluated")
    leaderboard = pd.DataFrame(rows)
    leaderboard["__idx__"] = range(len(leaderboard))
    leaderboard = leaderboard.sort_values(by=metric, ascending=False).reset_index(drop=True)
    best_idx = int(leaderboard["__idx__"].iloc[0])
    best_score = float(leaderboard[metric].iloc[0])
    leaderboard = leaderboard.drop(columns="__idx__")
    return GridSearchResult(
        best_params=param_sets[best_idx],
        best_score=best_score,
        leaderboard=leaderboard,
    )
