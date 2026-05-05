"""Market Overview page — fetch OHLCV and visualize with indicators."""

from __future__ import annotations

import streamlit as st

from app.dashboard.components import candlestick_chart
from app.data import fetch_ohlcv
from app.indicators import attach_default_indicators

st.set_page_config(page_title="Market Overview", page_icon="📊", layout="wide")
st.title("Market Overview")

with st.sidebar:
    st.header("Data")
    symbol = st.text_input("Symbol", value="BTC/USDT")
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
    limit = st.slider("Bars", min_value=100, max_value=2_000, value=500, step=50)
    exchange = st.selectbox("Exchange", ["binance", "kraken", "coinbase", "bybit"], index=0)
    show_sma = st.checkbox("SMA(20/50)", value=True)
    show_bb = st.checkbox("Bollinger Bands", value=False)
    fetch = st.button("Fetch", type="primary")

if fetch or "df" not in st.session_state:
    df = fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit, exchange=exchange)
    st.session_state["df"] = df
    st.session_state["symbol"] = symbol
    st.session_state["timeframe"] = timeframe

df = st.session_state["df"]
features = attach_default_indicators(df)

overlays = {}
if show_sma:
    overlays["SMA(20)"] = features["sma_20"]
    overlays["SMA(50)"] = features["sma_50"]
if show_bb:
    overlays["BB upper"] = features["bb_upper"]
    overlays["BB lower"] = features["bb_lower"]

st.plotly_chart(
    candlestick_chart(df, title=f"{symbol} — {timeframe}", overlays=overlays),
    use_container_width=True,
)

with st.expander("Latest indicator panel"):
    st.dataframe(features.tail(20).round(3), use_container_width=True)
