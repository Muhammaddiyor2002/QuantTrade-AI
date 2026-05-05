"""Backtesting page — run a strategy through the engine."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import streamlit as st

from app.backtesting import BacktestConfig, BacktestEngine
from app.config import get_settings
from app.dashboard.components import candlestick_chart, equity_chart, metrics_panel
from app.data import fetch_ohlcv
from app.strategies import get_strategy, list_strategies

st.set_page_config(page_title="Backtesting", page_icon="🧪", layout="wide")
st.title("Backtesting")

settings = get_settings()
with st.sidebar:
    st.header("Run")
    symbol = st.text_input("Symbol", value="BTC/USDT")
    timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=0)
    limit = st.slider("Bars", 300, 2_000, 1_000, 50)
    strategy = st.selectbox("Strategy", list_strategies())
    initial = st.number_input("Initial capital", value=settings.initial_capital, step=1_000.0)
    commission = st.number_input("Commission (bps)", value=settings.commission_bps, step=1.0)
    slippage = st.number_input("Slippage (bps)", value=settings.slippage_bps, step=1.0)
    fraction = st.slider("Equity fraction per trade", 0.1, 1.0, 1.0, 0.1)

df = fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
strat = get_strategy(strategy)
engine = BacktestEngine(
    BacktestConfig(
        initial_capital=initial,
        commission_bps=commission,
        slippage_bps=slippage,
        fraction=fraction,
        timeframe=timeframe,
    )
)
result = engine.run_strategy(df, strat)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total return", f"{result.metrics.total_return:.2%}")
c2.metric("Sharpe", f"{result.metrics.sharpe:.2f}")
c3.metric("Max DD", f"{result.metrics.max_drawdown:.2%}")
c4.metric("Win rate", f"{result.metrics.win_rate:.2%}")

st.plotly_chart(equity_chart(result.equity, title=f"{symbol} {strategy}"), use_container_width=True)

trades_df = pd.DataFrame([asdict(t) for t in result.trades])
st.plotly_chart(
    candlestick_chart(df, title="Trades", trades=trades_df),
    use_container_width=True,
)

st.subheader("Performance metrics")
st.dataframe(metrics_panel(result.metrics.to_dict()), use_container_width=True, hide_index=True)

if not trades_df.empty:
    st.subheader("Trade log")
    st.dataframe(trades_df.round(4), use_container_width=True)
