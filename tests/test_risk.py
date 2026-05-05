from __future__ import annotations

import pandas as pd

from app.risk.manager import (
    RiskConfig,
    apply_stops_and_targets,
    atr_stops,
    fixed_pct_stops,
    position_size,
)
from app.risk.portfolio import (
    correlation_matrix,
    cvar_historical,
    equal_weight_portfolio,
    kelly_fraction,
    risk_parity_weights,
    var_historical,
)
from app.strategies import MovingAverageCrossover


def test_position_size_basic() -> None:
    qty = position_size(10_000.0, 100.0, 95.0, risk_per_trade=0.01)
    assert qty == 20.0


def test_position_size_invalid_returns_zero() -> None:
    assert position_size(10_000.0, 100.0, 100.0) == 0.0
    assert position_size(0.0, 100.0, 95.0) == 0.0


def test_fixed_pct_stops_long() -> None:
    sl, tp = fixed_pct_stops(100.0, 1, sl_pct=0.02, tp_pct=0.04)
    assert sl == 98.0 and tp == 104.0


def test_atr_stops_short() -> None:
    sl, tp = atr_stops(100.0, 2.0, -1, multiplier=1.0, rr=2.0)
    assert sl > 100.0
    assert tp < 100.0


def test_apply_stops_keeps_signal_in_range(ohlcv_medium: pd.DataFrame) -> None:
    strat = MovingAverageCrossover(fast=10, slow=30)
    sig = strat.generate_signals(ohlcv_medium)
    risked = apply_stops_and_targets(ohlcv_medium, sig, RiskConfig(stop_loss_pct=0.01))
    assert risked.position.isin([-1, 0, 1]).all()


def test_apply_stops_does_not_immediately_reenter() -> None:
    """A tight stop on a downward bar should flatten the position and NOT
    re-enter on the same bar (regression for stale-target re-entry bug)."""
    import numpy as np
    import pandas as pd

    from app.strategies.base import Signals

    idx = pd.date_range("2024-01-01", periods=5, freq="1h")
    # Bar 0: setup; Bar 1: long entry; Bar 2: deep gap-down stop hit;
    # Bars 3-4: signal still long but stop has flattened.
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 80.0, 80.0, 80.0],
            "high": [101.0, 101.0, 80.5, 80.5, 80.5],
            "low": [99.0, 99.0, 79.5, 79.5, 79.5],
            "close": [100.0, 100.0, 80.0, 80.0, 80.0],
            "volume": [1, 1, 1, 1, 1],
        },
        index=idx,
    )
    raw = Signals(position=pd.Series([0, 1, 1, 1, 1], index=idx))
    risked = apply_stops_and_targets(df, raw, RiskConfig(stop_loss_pct=0.05))
    # On bar 2 the 5% stop fires; the bug would re-open immediately. Assert
    # that position[2:] is flattened (or at least not reverted to 1 on bar 2).
    assert int(risked.position.iloc[2]) == 0
    assert np.array_equal(risked.position.values, np.array([0, 1, 0, 0, 0]))


def test_var_cvar(ohlcv_medium: pd.DataFrame) -> None:
    rets = ohlcv_medium["close"].pct_change().dropna()
    v = var_historical(rets, 0.05)
    c = cvar_historical(rets, 0.05)
    assert v >= 0
    assert c >= v


def test_kelly_clipping() -> None:
    assert 0.0 <= kelly_fraction(0.05, 0.04) <= 1.0
    assert kelly_fraction(-0.1, 0.04) == 0.0
    assert kelly_fraction(1.0, 0.0) == 0.0


def test_portfolio_weights() -> None:
    rets = pd.DataFrame(
        {
            "A": [0.01, -0.005, 0.002],
            "B": [-0.002, 0.001, 0.003],
            "C": [0.005, 0.002, -0.001],
        }
    )
    eq = equal_weight_portfolio(rets)
    rp = risk_parity_weights(rets)
    corr = correlation_matrix(rets)
    assert abs(eq.sum() - 1.0) < 1e-9
    assert abs(rp.sum() - 1.0) < 1e-9
    assert corr.shape == (3, 3)
