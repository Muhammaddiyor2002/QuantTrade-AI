"""Command-line entrypoint for QuantTrade AI."""

from __future__ import annotations

import json

import typer

from app.backtesting import BacktestConfig, BacktestEngine
from app.config import get_settings
from app.data import fetch_ohlcv
from app.models import train_and_predict
from app.optimization import compare_strategies, grid_search, walk_forward
from app.strategies import get_strategy, list_strategies
from app.utils.logger import configure_logging, get_logger

cli = typer.Typer(help="QuantTrade AI command-line interface.", no_args_is_help=True)
configure_logging()
log = get_logger(__name__)


@cli.command()
def strategies() -> None:
    """List registered strategies."""
    typer.echo("\n".join(list_strategies()))


@cli.command()
def backtest(
    strategy: str = "ma_crossover",
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 1_000,
) -> None:
    """Run a single-strategy backtest and print metrics."""
    settings = get_settings()
    df = fetch_ohlcv(symbol, timeframe, limit)
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=settings.initial_capital,
            commission_bps=settings.commission_bps,
            slippage_bps=settings.slippage_bps,
            timeframe=timeframe,
        )
    )
    result = engine.run_strategy(df, get_strategy(strategy))
    typer.echo(json.dumps(result.metrics.to_dict(), indent=2))


@cli.command()
def predict(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 1_000) -> None:
    """Train an XGBoost predictor and emit the latest probability + signal."""
    df = fetch_ohlcv(symbol, timeframe, limit)
    _, report, pred = train_and_predict(df)
    typer.echo(
        json.dumps(
            {
                "cv": {"accuracy": report.accuracy, "auc": report.auc, "logloss": report.logloss},
                "last_proba_up": float(pred.proba.iloc[-1]),
                "last_signal": int(pred.signal.iloc[-1]),
            },
            indent=2,
        )
    )


@cli.command()
def optimize(
    strategy: str = "ma_crossover",
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 1_000,
) -> None:
    """Grid-search a small default param grid for the chosen strategy."""
    df = fetch_ohlcv(symbol, timeframe, limit)
    grids = {
        "ma_crossover": {"fast": [10, 20, 30], "slow": [50, 100, 150]},
        "rsi": {"period": [7, 14, 21], "oversold": [25, 30], "overbought": [65, 70, 75]},
        "breakout": {"entry_window": [10, 20, 40], "exit_window": [5, 10, 20]},
    }
    if strategy not in grids:
        raise typer.BadParameter(f"No default grid for {strategy}")
    res = grid_search(df, strategy, grids[strategy], config=BacktestConfig(timeframe=timeframe))
    typer.echo(json.dumps({"best": res.best_params, "score": res.best_score}, indent=2))


@cli.command()
def walkforward(
    strategy: str = "ma_crossover",
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 1_500,
    folds: int = 4,
) -> None:
    """Run a walk-forward analysis."""
    df = fetch_ohlcv(symbol, timeframe, limit)
    grids = {
        "ma_crossover": {"fast": [10, 20], "slow": [50, 100]},
        "rsi": {"period": [14, 21], "oversold": [25, 30], "overbought": [70]},
        "breakout": {"entry_window": [20, 40], "exit_window": [10, 20]},
    }
    rep = walk_forward(
        df, strategy, grids[strategy], n_folds=folds, config=BacktestConfig(timeframe=timeframe)
    )
    typer.echo(rep.summary.to_string(index=False))


@cli.command()
def compare(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 1_000) -> None:
    """Compare default presets of all built-in strategies."""
    df = fetch_ohlcv(symbol, timeframe, limit)
    pairs = [
        ("ma_crossover", {"fast": 20, "slow": 50}),
        ("rsi", {"period": 14, "oversold": 30, "overbought": 70}),
        ("breakout", {"entry_window": 20, "exit_window": 10}),
    ]
    df_out = compare_strategies(df, pairs, config=BacktestConfig(timeframe=timeframe))
    typer.echo(df_out.to_string(index=False))


if __name__ == "__main__":
    cli()
