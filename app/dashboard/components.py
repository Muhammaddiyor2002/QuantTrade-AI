"""Reusable Streamlit / Plotly components for the QuantTrade dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def candlestick_chart(
    df: pd.DataFrame,
    *,
    title: str = "Price",
    overlays: dict[str, pd.Series] | None = None,
    trades: pd.DataFrame | None = None,
    height: int = 600,
) -> go.Figure:
    """Build a candlestick chart with optional indicator overlays and trade markers."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.03,
        subplot_titles=(title, "Volume"),
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )
    if overlays:
        for name, series in overlays.items():
            fig.add_trace(
                go.Scatter(x=series.index, y=series.values, name=name, mode="lines"),
                row=1,
                col=1,
            )
    if trades is not None and not trades.empty:
        longs = trades[trades["direction"] == 1]
        shorts = trades[trades["direction"] == -1]
        if not longs.empty:
            fig.add_trace(
                go.Scatter(
                    x=longs["entry_time"],
                    y=longs["entry_price"],
                    mode="markers",
                    name="Long entry",
                    marker={"color": "#22c55e", "symbol": "triangle-up", "size": 10},
                ),
                row=1,
                col=1,
            )
        if not shorts.empty:
            fig.add_trace(
                go.Scatter(
                    x=shorts["entry_time"],
                    y=shorts["entry_price"],
                    mode="markers",
                    name="Short entry",
                    marker={"color": "#ef4444", "symbol": "triangle-down", "size": 10},
                ),
                row=1,
                col=1,
            )
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color="#64748b"),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=height,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        legend={"orientation": "h", "y": 1.02, "x": 0.01},
    )
    return fig


def equity_chart(equity: pd.Series, title: str = "Equity Curve", height: int = 400) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Equity", mode="lines"))
    running_max = equity.cummax()
    drawdown = (equity / running_max - 1.0) * 100
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name="Drawdown %",
            mode="lines",
            yaxis="y2",
            line={"color": "#ef4444"},
        )
    )
    fig.update_layout(
        title=title,
        height=height,
        yaxis={"title": "Equity"},
        yaxis2={"title": "Drawdown %", "overlaying": "y", "side": "right"},
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    return fig


def metrics_panel(metrics: dict[str, float]) -> pd.DataFrame:
    fmt: dict[str, str] = {
        "total_return": "{:.2%}",
        "cagr": "{:.2%}",
        "sharpe": "{:.2f}",
        "sortino": "{:.2f}",
        "max_drawdown": "{:.2%}",
        "calmar": "{:.2f}",
        "win_rate": "{:.2%}",
        "profit_factor": "{:.2f}",
        "exposure": "{:.2%}",
    }
    rows = [{"metric": k, "value": fmt.get(k, "{:.4f}").format(v)} for k, v in metrics.items()]
    return pd.DataFrame(rows)
