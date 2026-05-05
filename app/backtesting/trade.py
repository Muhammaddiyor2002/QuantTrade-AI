"""Trade representation."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class Trade:
    """A single closed trade."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int  # +1 long, -1 short
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    bars_held: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0
