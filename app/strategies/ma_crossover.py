"""Classic moving-average crossover strategy."""

from __future__ import annotations

import pandas as pd

from app.indicators.trend import ema, sma
from app.strategies.base import Signals, Strategy


class MovingAverageCrossover(Strategy):
    """Long when fast MA > slow MA, flat otherwise (long-only by default)."""

    name = "ma_crossover"

    def __init__(
        self,
        fast: int = 20,
        slow: int = 50,
        ma_type: str = "sma",
        allow_short: bool = False,
    ) -> None:
        if fast >= slow:
            raise ValueError("fast period must be < slow period")
        super().__init__(fast=fast, slow=slow, ma_type=ma_type, allow_short=allow_short)

    def generate_signals(self, df: pd.DataFrame) -> Signals:
        if "close" not in df.columns:
            raise ValueError("df must contain a 'close' column")
        ma_fn = ema if self.params["ma_type"] == "ema" else sma
        fast_ma = ma_fn(df["close"], self.params["fast"])
        slow_ma = ma_fn(df["close"], self.params["slow"])

        long_pos = (fast_ma > slow_ma).astype("int64")
        if self.params["allow_short"]:
            short_pos = (fast_ma < slow_ma).astype("int64") * -1
            position = (long_pos + short_pos).astype("int64")
        else:
            position = long_pos

        position = position.fillna(0).astype("int64")
        position.name = "position"
        return Signals(
            position=position,
            metadata={"fast_ma": fast_ma, "slow_ma": slow_ma},
        )
