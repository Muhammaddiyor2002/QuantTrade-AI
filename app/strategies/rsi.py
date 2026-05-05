"""RSI mean-reversion strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.indicators.momentum import rsi as rsi_indicator
from app.strategies.base import Signals, Strategy


class RSIStrategy(Strategy):
    """Buy oversold, sell overbought (long-only by default).

    Position is long while last entry signal hasn't been closed by an exit.
    """

    name = "rsi"

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        allow_short: bool = False,
    ) -> None:
        if not 0 < oversold < overbought < 100:
            raise ValueError("require 0 < oversold < overbought < 100")
        super().__init__(
            period=period,
            oversold=oversold,
            overbought=overbought,
            allow_short=allow_short,
        )

    def generate_signals(self, df: pd.DataFrame) -> Signals:
        rsi_series = rsi_indicator(df["close"], self.params["period"])
        oversold = self.params["oversold"]
        overbought = self.params["overbought"]

        n = len(df)
        position = np.zeros(n, dtype=np.int64)
        rsi_arr = rsi_series.to_numpy()
        long_short = self.params["allow_short"]

        current = 0
        for i in range(n):
            r = rsi_arr[i]
            if np.isnan(r):
                position[i] = current
                continue
            if current == 0:
                if r < oversold:
                    current = 1
                elif long_short and r > overbought:
                    current = -1
            elif current == 1 and r > overbought:
                current = 0
            elif current == -1 and r < oversold:
                current = 0
            position[i] = current

        pos_series = pd.Series(position, index=df.index, name="position")
        return Signals(position=pos_series, metadata={"rsi": rsi_series})
