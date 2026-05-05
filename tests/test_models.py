from __future__ import annotations

import pandas as pd

from app.models.features import build_xy, make_features, make_target
from app.models.pipeline import signals_from_proba


def test_make_features_columns(ohlcv_medium: pd.DataFrame) -> None:
    feats = make_features(ohlcv_medium)
    assert "ret_1" in feats.columns
    assert "rsi_14" in feats.columns
    assert "macd" in feats.columns


def test_make_target_binary(ohlcv_medium: pd.DataFrame) -> None:
    y = make_target(ohlcv_medium["close"], horizon=1)
    assert y.isin([0, 1]).all()


def test_build_xy_alignment(ohlcv_medium: pd.DataFrame) -> None:
    X, y = build_xy(ohlcv_medium)
    assert len(X) == len(y)
    assert X.notna().all().all()


def test_signals_from_proba_thresholds(ohlcv_medium: pd.DataFrame) -> None:
    proba = pd.Series([0.1, 0.4, 0.5, 0.6, 0.9])
    pos = signals_from_proba(proba, long_threshold=0.6, short_threshold=0.4, allow_short=True)
    assert pos.iloc[-1] == 1
    assert pos.iloc[0] == -1
    assert pos.iloc[2] == 0
