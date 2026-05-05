"""Donchian-channel breakout (Turtle-style) strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.base import Signals, Strategy


class BreakoutStrategy(Strategy):
    """Long on close > rolling N-period high; flat on close < rolling exit-low."""

    name = "breakout"

    def __init__(
        self,
        entry_window: int = 20,
        exit_window: int = 10,
        allow_short: bool = False,
    ) -> None:
        if entry_window <= exit_window:
            raise ValueError("entry_window must be greater than exit_window")
        super().__init__(
            entry_window=entry_window,
            exit_window=exit_window,
            allow_short=allow_short,
        )

    def generate_signals(self, df: pd.DataFrame) -> Signals:
        entry = self.params["entry_window"]
        exit_w = self.params["exit_window"]
        high = df["high"]
        low = df["low"]
        close = df["close"]

        upper = high.shift(1).rolling(entry, min_periods=entry).max()
        lower = low.shift(1).rolling(entry, min_periods=entry).min()
        exit_low = low.shift(1).rolling(exit_w, min_periods=exit_w).min()
        exit_high = high.shift(1).rolling(exit_w, min_periods=exit_w).max()

        n = len(df)
        position = np.zeros(n, dtype=np.int64)
        c = close.to_numpy()
        u = upper.to_numpy()
        ll = lower.to_numpy()
        xl = exit_low.to_numpy()
        xh = exit_high.to_numpy()
        long_short = self.params["allow_short"]

        current = 0
        for i in range(n):
            if np.isnan(u[i]) or np.isnan(ll[i]):
                position[i] = current
                continue
            if current == 0:
                if c[i] > u[i]:
                    current = 1
                elif long_short and c[i] < ll[i]:
                    current = -1
            elif current == 1 and not np.isnan(xl[i]) and c[i] < xl[i]:
                current = 0
            elif current == -1 and not np.isnan(xh[i]) and c[i] > xh[i]:
                current = 0
            position[i] = current

        pos_series = pd.Series(position, index=df.index, name="position")
        return Signals(
            position=pos_series,
            metadata={"upper": upper, "lower": lower},
        )
