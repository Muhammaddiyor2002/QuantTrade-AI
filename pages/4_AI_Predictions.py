"""AI Predictions page — train an XGBoost model and inspect signals."""

from __future__ import annotations

import streamlit as st

from app.backtesting import BacktestConfig, BacktestEngine
from app.dashboard.components import candlestick_chart, equity_chart
from app.data import fetch_ohlcv
from app.models import XGBConfig, feature_importance, train_and_predict
from app.strategies.base import Signals

st.set_page_config(page_title="AI Predictions", page_icon="🤖", layout="wide")
st.title("AI Predictions")

with st.sidebar:
    st.header("Training")
    symbol = st.text_input("Symbol", value="BTC/USDT")
    timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=0)
    limit = st.slider("Bars", 500, 3_000, 1_500, 100)
    horizon = st.slider("Forecast horizon (bars)", 1, 10, 1)
    n_estimators = st.slider("XGB n_estimators", 50, 800, 300, 50)
    max_depth = st.slider("XGB max_depth", 2, 8, 4)
    long_threshold = st.slider("Long threshold", 0.50, 0.80, 0.55, 0.01)
    short_threshold = st.slider("Short threshold", 0.20, 0.50, 0.45, 0.01)
    allow_short = st.checkbox("Allow short", value=False)

df = fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
config = XGBConfig(n_estimators=n_estimators, max_depth=max_depth)
model, report, pred = train_and_predict(
    df,
    horizon=horizon,
    config=config,
    long_threshold=long_threshold,
    short_threshold=short_threshold,
    allow_short=allow_short,
)

c1, c2, c3 = st.columns(3)
c1.metric("CV accuracy", f"{report.accuracy:.3f}")
c2.metric("CV AUC", f"{report.auc:.3f}")
c3.metric("Samples", report.n_samples)

st.subheader("Probability of UP")
st.line_chart(pred.proba.rename("p(up)"))

st.subheader("Feature importance")
st.bar_chart(feature_importance(model).head(15))

engine = BacktestEngine(BacktestConfig(timeframe=timeframe))
signals = Signals(position=pred.signal.astype(int), metadata={"proba_up": pred.proba})
result = engine.run(df, signals)
st.plotly_chart(
    equity_chart(result.equity, title=f"AI strategy on {symbol}"),
    use_container_width=True,
)
st.plotly_chart(
    candlestick_chart(df, title="Predictions", overlays={"p(up)": pred.proba}),
    use_container_width=True,
)
