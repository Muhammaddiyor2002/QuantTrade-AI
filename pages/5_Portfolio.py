"""Portfolio Tracker — multi-strategy comparison."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.backtesting import BacktestConfig, BacktestEngine
from app.data import fetch_ohlcv
from app.strategies import get_strategy, list_strategies

st.set_page_config(page_title="Portfolio Tracker", page_icon="📒", layout="wide")
st.title("Portfolio Tracker")

with st.sidebar:
    st.header("Universe")
    symbols = st.text_area(
        "Symbols (comma-separated)",
        value="BTC/USDT, ETH/USDT, SOL/USDT",
    )
    timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=2)
    limit = st.slider("Bars", 300, 2_000, 800, 50)
    strategy = st.selectbox("Strategy", list_strategies())

symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
engine = BacktestEngine(BacktestConfig(timeframe=timeframe))
records = []
equity_curves: dict[str, pd.Series] = {}
for symbol in symbol_list:
    df = fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
    result = engine.run_strategy(df, get_strategy(strategy))
    equity_curves[symbol] = (result.equity / result.equity.iloc[0] - 1.0) * 100
    records.append({"symbol": symbol, **result.metrics.to_dict()})

st.subheader("Per-asset metrics")
st.dataframe(pd.DataFrame(records).round(4), use_container_width=True, hide_index=True)

st.subheader("Equity curves (% return)")
chart_df = pd.concat(equity_curves, axis=1)
st.line_chart(chart_df)
