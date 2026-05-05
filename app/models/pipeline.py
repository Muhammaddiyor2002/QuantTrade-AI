"""End-to-end ML pipeline: features -> train -> predict -> trade signal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.models.features import build_xy, make_features
from app.models.xgb_model import TrainReport, XGBConfig, XGBPriceModel
from app.strategies.base import Signals
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class PredictionResult:
    proba: pd.Series
    prediction: pd.Series
    signal: pd.Series  # in {-1, 0, 1}


def signals_from_proba(
    proba: pd.Series,
    *,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    allow_short: bool = False,
) -> pd.Series:
    """Convert a probability-of-up Series into integer positions."""
    pos = pd.Series(0, index=proba.index, dtype="int64")
    pos[proba >= long_threshold] = 1
    if allow_short:
        pos[proba <= short_threshold] = -1
    return pos


def train_and_predict(
    df: pd.DataFrame,
    *,
    horizon: int = 1,
    config: XGBConfig | None = None,
    n_splits: int = 5,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    allow_short: bool = False,
) -> tuple[XGBPriceModel, TrainReport, PredictionResult]:
    """Train an XGBoost model and produce in-sample probability + trading signals."""
    X, y = build_xy(df, horizon=horizon)
    model = XGBPriceModel(config)
    if len(X) > n_splits * 2:
        report = model.cross_validate(X, y, n_splits=n_splits)
    else:
        report = TrainReport(
            accuracy=float("nan"),
            auc=float("nan"),
            logloss=float("nan"),
            n_samples=len(X),
            n_features=X.shape[1],
            folds=0,
        )
    model.fit(X, y)
    proba_up = model.predict_proba(X)[:, 1]
    proba = pd.Series(proba_up, index=X.index, name="proba_up")
    pred = pd.Series((proba_up > 0.5).astype(int), index=X.index, name="prediction")

    full_index = make_features(df).index
    proba_full = proba.reindex(full_index).ffill().fillna(0.5)
    signal = signals_from_proba(
        proba_full,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        allow_short=allow_short,
    ).rename("signal")
    return (
        model,
        report,
        PredictionResult(
            proba=proba_full,
            prediction=pred.reindex(full_index).ffill().fillna(0).astype(int),
            signal=signal,
        ),
    )


def predict_to_signals(model: XGBPriceModel, df: pd.DataFrame, **kw) -> Signals:
    """Apply a fitted model to OHLCV and return :class:`Signals`."""
    feats = make_features(df)
    X = feats[model.feature_columns].dropna()
    proba_up = model.predict_proba(X)[:, 1]
    proba = pd.Series(proba_up, index=X.index, name="proba_up")
    proba_full = proba.reindex(feats.index).ffill().fillna(0.5)
    pos = signals_from_proba(proba_full, **kw)
    pos = pos.reindex(df.index).fillna(0).astype(int)
    return Signals(position=pos, metadata={"proba_up": proba_full})


def feature_importance(model: XGBPriceModel) -> pd.Series:
    if model.model is None:
        raise RuntimeError("Model not trained.")
    importances = np.asarray(model.model.feature_importances_)
    return pd.Series(importances, index=model.feature_columns, name="importance").sort_values(
        ascending=False
    )
