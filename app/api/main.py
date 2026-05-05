"""FastAPI service exposing core QuantTrade AI capabilities."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.backtesting import BacktestConfig, BacktestEngine
from app.config import get_settings
from app.data import fetch_ohlcv
from app.models import train_and_predict
from app.optimization import compare_strategies
from app.strategies import get_strategy, list_strategies

app = FastAPI(
    title="QuantTrade AI API",
    description="Backtesting + AI prediction service for QuantTrade AI.",
    version="0.1.0",
)


class BacktestRequest(BaseModel):
    symbol: str = Field(default="BTC/USDT")
    timeframe: str = Field(default="1h")
    limit: int = Field(default=500, ge=50, le=5_000)
    strategy: str = Field(default="ma_crossover")
    params: dict[str, Any] = Field(default_factory=dict)
    initial_capital: float = 10_000.0
    commission_bps: float = 5.0
    slippage_bps: float = 2.0


class CompareRequest(BaseModel):
    symbol: str = Field(default="BTC/USDT")
    timeframe: str = Field(default="1h")
    limit: int = Field(default=500, ge=50, le=5_000)
    strategies: list[dict] = Field(default_factory=list)


class AIRequest(BaseModel):
    symbol: str = Field(default="BTC/USDT")
    timeframe: str = Field(default="1h")
    limit: int = Field(default=1_000, ge=200, le=5_000)
    horizon: int = 1


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "quanttrade-ai"}


@app.get("/strategies")
def get_strategies() -> dict[str, list[str]]:
    return {"strategies": list_strategies()}


@app.get("/config")
def get_config() -> dict[str, Any]:
    s = get_settings()
    return {
        "exchange": s.exchange,
        "default_symbol": s.default_symbol,
        "default_timeframe": s.default_timeframe,
        "initial_capital": s.initial_capital,
    }


@app.post("/backtest")
def backtest(req: BacktestRequest) -> dict[str, Any]:
    if req.strategy not in list_strategies():
        raise HTTPException(404, f"Unknown strategy: {req.strategy}")
    df = fetch_ohlcv(req.symbol, req.timeframe, req.limit)
    strat = get_strategy(req.strategy, **req.params)
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=req.initial_capital,
            commission_bps=req.commission_bps,
            slippage_bps=req.slippage_bps,
            timeframe=req.timeframe,
        )
    )
    result = engine.run_strategy(df, strat)
    return {
        "metrics": result.metrics.to_dict(),
        "num_trades": len(result.trades),
        "final_equity": float(result.equity.iloc[-1]),
    }


@app.post("/compare")
def compare(req: CompareRequest) -> dict[str, Any]:
    df = fetch_ohlcv(req.symbol, req.timeframe, req.limit)
    pairs: list[tuple[str, dict]] = []
    for s in req.strategies:
        name = s.get("name")
        if not isinstance(name, str):
            raise HTTPException(400, "Each strategy entry must include a string 'name'.")
        pairs.append((name, dict(s.get("params") or {})))
    df_out = compare_strategies(df, pairs, config=BacktestConfig(timeframe=req.timeframe))
    return {"leaderboard": df_out.to_dict(orient="records")}


@app.post("/predict")
def predict(req: AIRequest) -> dict[str, Any]:
    df = fetch_ohlcv(req.symbol, req.timeframe, req.limit)
    _, report, pred = train_and_predict(df, horizon=req.horizon)
    return {
        "report": report.__dict__,
        "last_proba_up": float(pred.proba.iloc[-1]),
        "last_signal": int(pred.signal.iloc[-1]),
    }
