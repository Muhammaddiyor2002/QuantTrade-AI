"""Walk-forward analysis (anchored / rolling)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.backtesting import BacktestConfig, BacktestEngine, BacktestResult
from app.optimization.grid_search import grid_search
from app.strategies import get_strategy
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class WalkForwardFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict
    in_sample_score: float
    out_of_sample: BacktestResult


@dataclass(slots=True)
class WalkForwardReport:
    folds: list[WalkForwardFold]
    summary: pd.DataFrame
    aggregated_equity: pd.Series


def walk_forward(
    df: pd.DataFrame,
    strategy_name: str,
    grid: dict[str, list],
    *,
    n_folds: int = 5,
    train_size: float = 0.6,
    config: BacktestConfig | None = None,
    metric: str = "sharpe",
) -> WalkForwardReport:
    """Run anchored walk-forward optimization on `df`.

    Each fold tunes hyperparameters on the training window, then evaluates the
    next contiguous slice out-of-sample.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    n = len(df)
    train_n = int(n * train_size)
    test_n = (n - train_n) // n_folds
    if test_n <= 0:
        raise ValueError("Not enough rows for the requested folds.")
    cfg = config or BacktestConfig()

    folds: list[WalkForwardFold] = []
    equities: list[pd.Series] = []

    for k in range(n_folds):
        train_start = 0
        train_end = train_n + k * test_n
        test_start = train_end
        test_end = min(test_start + test_n, n)
        if test_end <= test_start:
            break
        train = df.iloc[train_start:train_end]
        test = df.iloc[test_start:test_end]

        gs = grid_search(train, strategy_name, grid, config=cfg, metric=metric)
        engine = BacktestEngine(cfg)
        oos = engine.run_strategy(test, get_strategy(strategy_name, **gs.best_params))
        folds.append(
            WalkForwardFold(
                train_start=train.index[0],
                train_end=train.index[-1],
                test_start=test.index[0],
                test_end=test.index[-1],
                best_params=gs.best_params,
                in_sample_score=gs.best_score,
                out_of_sample=oos,
            )
        )
        equities.append(oos.equity)

    if not equities:
        raise RuntimeError("Walk-forward produced no folds.")

    aggregated = pd.concat(equities)
    aggregated = aggregated[~aggregated.index.duplicated(keep="last")].sort_index()

    summary = pd.DataFrame(
        [
            {
                "fold": i,
                "train_start": f.train_start,
                "train_end": f.train_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
                "is_score": f.in_sample_score,
                "oos_sharpe": f.out_of_sample.metrics.sharpe,
                "oos_return": f.out_of_sample.metrics.total_return,
                "oos_mdd": f.out_of_sample.metrics.max_drawdown,
                **{f"param_{k}": v for k, v in f.best_params.items()},
            }
            for i, f in enumerate(folds)
        ]
    )
    log.info(
        "Walk-forward: %d folds | mean OOS Sharpe = %.2f",
        len(folds),
        float(np.nanmean([f.out_of_sample.metrics.sharpe for f in folds])),
    )
    return WalkForwardReport(folds=folds, summary=summary, aggregated_equity=aggregated)
