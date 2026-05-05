"""Strategy Builder — pick a strategy and inspect generated signals."""

from __future__ import annotations

import streamlit as st

from app.dashboard.components import candlestick_chart
from app.data import fetch_ohlcv
from app.strategies import get_strategy, list_strategies

st.set_page_config(page_title="Strategy Builder", page_icon="🛠", layout="wide")
st.title("Strategy Builder")

with st.sidebar:
    st.header("Inputs")
    symbol = st.text_input("Symbol", value="BTC/USDT")
    timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)
    limit = st.slider("Bars", 200, 2_000, 800, 50)

    name = st.selectbox("Strategy", list_strategies())
    params: dict[str, object] = {}
    if name == "ma_crossover":
        params["fast"] = st.slider("Fast MA", 5, 60, 20)
        params["slow"] = st.slider("Slow MA", 30, 200, 50)
        params["ma_type"] = st.radio("MA type", ["sma", "ema"], horizontal=True)
        params["allow_short"] = st.checkbox("Allow short", value=False)
    elif name == "rsi":
        params["period"] = st.slider("RSI period", 5, 30, 14)
        params["oversold"] = st.slider("Oversold", 5.0, 45.0, 30.0)
        params["overbought"] = st.slider("Overbought", 55.0, 95.0, 70.0)
        params["allow_short"] = st.checkbox("Allow short", value=False)
    elif name == "breakout":
        params["entry_window"] = st.slider("Entry window", 10, 60, 20)
        params["exit_window"] = st.slider("Exit window", 5, 30, 10)
        params["allow_short"] = st.checkbox("Allow short", value=False)

df = fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
strat = get_strategy(name, **params)
signals = strat.generate_signals(df)

overlays = {}
for k, v in signals.metadata.items():
    if hasattr(v, "index"):
        overlays[k] = v

st.plotly_chart(
    candlestick_chart(df, title=f"{symbol} — {strat}", overlays=overlays),
    use_container_width=True,
)
st.metric("In-market exposure", f"{signals.position.ne(0).mean():.1%}")
st.line_chart(signals.position.rename("position"))
