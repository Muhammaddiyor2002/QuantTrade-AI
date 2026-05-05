"""Risk Analysis — stops, VaR/CVaR, position sizing helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.backtesting import BacktestConfig, BacktestEngine
from app.config import get_settings
from app.dashboard.components import equity_chart
from app.data import fetch_ohlcv
from app.risk import (
    RiskConfig,
    apply_stops_and_targets,
    cvar_historical,
    fixed_pct_stops,
    position_size,
    var_historical,
)
from app.strategies import get_strategy, list_strategies

st.set_page_config(page_title="Risk Analysis", page_icon="🛡", layout="wide")
st.title("Risk Analysis")

settings = get_settings()
with st.sidebar:
    st.header("Risk parameters")
    sl = st.slider("Stop-loss %", 0.005, 0.10, 0.02, 0.005)
    tp = st.slider("Take-profit %", 0.005, 0.20, 0.04, 0.005)
    risk_per_trade = st.slider("Risk per trade (frac)", 0.001, 0.05, settings.risk_per_trade, 0.001)
    use_atr = st.checkbox("ATR-based stops", value=False)

    st.header("Run")
    symbol = st.text_input("Symbol", value="BTC/USDT")
    timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=0)
    limit = st.slider("Bars", 300, 2_000, 800, 50)
    strategy = st.selectbox("Strategy", list_strategies())

df = fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
strat = get_strategy(strategy)
signals = strat.generate_signals(df)

cfg = RiskConfig(
    risk_per_trade=risk_per_trade,
    stop_loss_pct=sl,
    take_profit_pct=tp,
    use_atr_stops=use_atr,
)
risked = apply_stops_and_targets(df, signals, cfg)

engine = BacktestEngine(BacktestConfig(timeframe=timeframe))
plain = engine.run(df, signals)
gated = engine.run(df, risked)

c1, c2, c3 = st.columns(3)
c1.metric("VaR(95%)", f"{var_historical(plain.returns, 0.05):.2%}")
c2.metric("CVaR(95%)", f"{cvar_historical(plain.returns, 0.05):.2%}")
c3.metric("Max DD (no stops)", f"{plain.metrics.max_drawdown:.2%}")

st.plotly_chart(equity_chart(plain.equity, title="Without stops"), use_container_width=True)
st.plotly_chart(equity_chart(gated.equity, title="With stops"), use_container_width=True)

st.subheader("Position-size example")
last_close = float(df["close"].iloc[-1])
sl_price, tp_price = fixed_pct_stops(last_close, 1, sl_pct=sl, tp_pct=tp)
qty = position_size(settings.initial_capital, last_close, sl_price, risk_per_trade=risk_per_trade)
st.dataframe(
    pd.DataFrame(
        [
            {"metric": "Last close", "value": f"{last_close:,.2f}"},
            {"metric": "Stop loss", "value": f"{sl_price:,.2f}"},
            {"metric": "Take profit", "value": f"{tp_price:,.2f}"},
            {"metric": "Suggested quantity", "value": f"{qty:,.4f}"},
        ]
    ),
    use_container_width=True,
    hide_index=True,
)
