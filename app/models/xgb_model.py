"""XGBoost-based price-direction classifier."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

try:  # pragma: no cover - import-time only
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
    raise ImportError("xgboost is required for app.models.xgb_model") from exc

from app.models.features import FEATURE_COLUMNS, build_xy
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class XGBConfig:
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    random_state: int = 42
    eval_metric: str = "logloss"


@dataclass(slots=True)
class TrainReport:
    accuracy: float
    auc: float
    logloss: float
    n_samples: int
    n_features: int
    folds: int


class XGBPriceModel:
    """Wraps an :class:`xgboost.XGBClassifier` with sane defaults for OHLCV data."""

    def __init__(self, config: XGBConfig | None = None) -> None:
        self.config = config or XGBConfig()
        self.model: XGBClassifier | None = None
        self.feature_columns: list[str] = list(FEATURE_COLUMNS)

    def _build(self) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            eval_metric=self.config.eval_metric,
            tree_method="hist",
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> XGBPriceModel:
        self.feature_columns = list(X.columns)
        self.model = self._build()
        self.model.fit(X.values, y.values)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit() or load() first.")
        return self.model.predict_proba(X[self.feature_columns].values)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not trained.")
        return self.model.predict(X[self.feature_columns].values)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> TrainReport:
        accs: list[float] = []
        aucs: list[float] = []
        losses: list[float] = []
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            mdl = self._build().fit(X_tr.values, y_tr.values)
            preds = mdl.predict(X_te.values)
            proba = mdl.predict_proba(X_te.values)[:, 1]
            accs.append(float(accuracy_score(y_te, preds)))
            try:
                aucs.append(float(roc_auc_score(y_te, proba)))
            except ValueError:
                aucs.append(float("nan"))
            losses.append(float(log_loss(y_te, proba, labels=[0, 1])))
        report = TrainReport(
            accuracy=float(np.nanmean(accs)),
            auc=float(np.nanmean(aucs)),
            logloss=float(np.nanmean(losses)),
            n_samples=len(X),
            n_features=X.shape[1],
            folds=n_splits,
        )
        log.info(
            "CV: acc=%.3f auc=%.3f logloss=%.3f over %d folds",
            report.accuracy,
            report.auc,
            report.logloss,
            n_splits,
        )
        return report

    def save(self, path: str | Path) -> Path:
        if self.model is None:
            raise RuntimeError("No trained model to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "config": asdict(self.config),
            "feature_columns": self.feature_columns,
            "model": self.model,
        }
        joblib.dump(bundle, path)
        sidecar = path.with_suffix(".meta.json")
        sidecar.write_text(
            json.dumps(
                {"config": asdict(self.config), "feature_columns": self.feature_columns},
                indent=2,
            )
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> XGBPriceModel:
        bundle: dict[str, Any] = joblib.load(path)
        config_data = bundle.get("config") or {}
        instance = cls(XGBConfig(**config_data))
        instance.feature_columns = list(bundle["feature_columns"])
        instance.model = bundle["model"]
        return instance


def train_from_ohlcv(
    df: pd.DataFrame,
    *,
    horizon: int = 1,
    config: XGBConfig | None = None,
    n_splits: int = 5,
) -> tuple[XGBPriceModel, TrainReport]:
    """High-level helper: features -> CV -> fit on full data."""
    X, y = build_xy(df, horizon=horizon)
    model = XGBPriceModel(config)
    report = (
        model.cross_validate(X, y, n_splits=n_splits)
        if len(X) > n_splits * 2
        else (
            TrainReport(
                accuracy=float("nan"),
                auc=float("nan"),
                logloss=float("nan"),
                n_samples=len(X),
                n_features=X.shape[1],
                folds=0,
            )
        )
    )
    model.fit(X, y)
    return model, report
