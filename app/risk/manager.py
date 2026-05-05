"""Per-trade risk controls: stop-loss, take-profit, position sizing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.indicators.volatility import atr
from app.strategies.base import Signals
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class RiskConfig:
    risk_per_trade: float = 0.01  # fraction of equity risked per trade
    max_portfolio_risk: float = 0.05  # cumulative open risk cap
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    atr_stop_multiplier: float = 2.0
    use_atr_stops: bool = False
    max_leverage: float = 1.0


def position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    *,
    risk_per_trade: float = 0.01,
) -> float:
    """Volatility-aware position size (units of asset).

    Risks at most ``risk_per_trade * equity`` between entry and stop. Returns 0
    if the stop is invalid.
    """
    if entry_price <= 0 or risk_per_trade <= 0:
        return 0.0
    risk_per_unit = abs(entry_price - stop_price)
    if risk_per_unit <= 0:
        return 0.0
    return float((equity * risk_per_trade) / risk_per_unit)


def fixed_pct_stops(
    entry_price: float, direction: int, *, sl_pct: float, tp_pct: float
) -> tuple[float, float]:
    """Return ``(stop_loss_price, take_profit_price)`` for a fixed-percent rule."""
    if direction == 1:
        return entry_price * (1.0 - sl_pct), entry_price * (1.0 + tp_pct)
    if direction == -1:
        return entry_price * (1.0 + sl_pct), entry_price * (1.0 - tp_pct)
    return entry_price, entry_price


def atr_stops(
    entry_price: float,
    atr_value: float,
    direction: int,
    *,
    multiplier: float = 2.0,
    rr: float = 2.0,
) -> tuple[float, float]:
    """ATR-based stops with a reward:risk multiple."""
    risk = atr_value * multiplier
    if direction == 1:
        return entry_price - risk, entry_price + risk * rr
    if direction == -1:
        return entry_price + risk, entry_price - risk * rr
    return entry_price, entry_price


def apply_stops_and_targets(
    df: pd.DataFrame,
    signals: Signals,
    config: RiskConfig | None = None,
) -> Signals:
    """Augment a position series with intra-trade SL/TP that close the position.

    The result mutates the position column to flatten when a stop or target is
    hit on the *current bar*, evaluated using the current bar's high/low. The
    new entry price is taken from the close of the bar where position changed.
    """
    cfg = config or RiskConfig()
    pos = signals.position.copy()
    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    new_pos = np.array(pos.to_numpy(), copy=True)

    if cfg.use_atr_stops:
        atr_series = atr(df["high"], df["low"], df["close"]).to_numpy()
    else:
        atr_series = np.zeros(len(df))

    cur = 0
    entry_price = 0.0
    sl = tp = 0.0
    for i in range(len(df)):
        target = int(new_pos[i])

        if cur != 0 and (highs[i] >= max(sl, tp) or lows[i] <= min(sl, tp)):
            if cur == 1:
                if lows[i] <= sl or highs[i] >= tp:
                    new_pos[i:] = np.where(new_pos[i:] == 1, 0, new_pos[i:])
                    cur = 0
            elif cur == -1:
                if highs[i] >= sl or lows[i] <= tp:
                    new_pos[i:] = np.where(new_pos[i:] == -1, 0, new_pos[i:])
                    cur = 0

        if target != cur and target != 0:
            entry_price = closes[i]
            if cfg.use_atr_stops and atr_series[i] > 0:
                sl, tp = atr_stops(
                    entry_price,
                    atr_series[i],
                    target,
                    multiplier=cfg.atr_stop_multiplier,
                    rr=cfg.take_profit_pct / max(cfg.stop_loss_pct, 1e-9),
                )
            else:
                sl, tp = fixed_pct_stops(
                    entry_price, target, sl_pct=cfg.stop_loss_pct, tp_pct=cfg.take_profit_pct
                )
            cur = target
        elif target == 0 and cur != 0:
            cur = 0

    out_pos = pd.Series(new_pos.astype(int), index=df.index, name="position")
    return Signals(position=out_pos, metadata={**signals.metadata, "risk_config": cfg})
