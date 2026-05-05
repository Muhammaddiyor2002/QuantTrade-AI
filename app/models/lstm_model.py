"""Optional Keras/Torch-style LSTM placeholder.

PyTorch / Keras are heavy dependencies and we keep them optional. This
module exposes ``LSTMPriceModel`` that uses Keras if available, otherwise
falls back to an :class:`sklearn.linear_model.LogisticRegression` on the
flattened feature window so the API still works without a deep-learning
stack installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.models.features import FEATURE_COLUMNS
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class LSTMConfig:
    sequence_length: int = 32
    epochs: int = 5
    batch_size: int = 64
    units: int = 32
    dropout: float = 0.2
    learning_rate: float = 1e-3


def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= seq_len:
        raise ValueError(f"Not enough rows ({len(X)}) for sequence_length={seq_len}")
    n = len(X) - seq_len
    out_x = np.empty((n, seq_len, X.shape[1]), dtype=float)
    out_y = np.empty(n, dtype=float)
    for i in range(n):
        out_x[i] = X[i : i + seq_len]
        out_y[i] = y[i + seq_len]
    return out_x, out_y


class LSTMPriceModel:
    """LSTM-style sequence classifier (Keras if available, fallback otherwise)."""

    def __init__(self, config: LSTMConfig | None = None) -> None:
        self.config = config or LSTMConfig()
        self.feature_columns: list[str] = list(FEATURE_COLUMNS)
        self._model: Any = None
        self._fallback = False

    def _build_keras(self, n_features: int):  # pragma: no cover - optional
        from tensorflow import keras

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.config.sequence_length, n_features)),
                keras.layers.LSTM(self.config.units, return_sequences=False),
                keras.layers.Dropout(self.config.dropout),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LSTMPriceModel:
        self.feature_columns = list(X.columns)
        X_arr = X.to_numpy(dtype=float)
        y_arr = y.to_numpy(dtype=float)

        try:  # pragma: no cover - tensorflow optional
            from tensorflow import keras  # noqa: F401

            x_seq, y_seq = _make_sequences(X_arr, y_arr, self.config.sequence_length)
            mdl = self._build_keras(X_arr.shape[1])
            mdl.fit(
                x_seq,
                y_seq,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                verbose=0,
            )
            self._model = mdl
            self._fallback = False
            return self
        except Exception as exc:
            log.warning("Keras unavailable (%s); falling back to LogisticRegression.", exc)

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        x_seq, y_seq = _make_sequences(X_arr, y_arr, self.config.sequence_length)
        flat = x_seq.reshape(x_seq.shape[0], -1)
        scaler = StandardScaler().fit(flat)
        clf = LogisticRegression(max_iter=2_000)
        clf.fit(scaler.transform(flat), y_seq)
        self._model = (scaler, clf)
        self._fallback = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not trained.")
        X_arr = X[self.feature_columns].to_numpy(dtype=float)
        seq_len = self.config.sequence_length
        if len(X_arr) <= seq_len:
            raise ValueError(f"Need >{seq_len} rows for prediction.")
        x_seq, _ = _make_sequences(X_arr, np.zeros(len(X_arr)), seq_len)
        model = self._model
        if self._fallback:
            scaler, clf = model
            flat = x_seq.reshape(x_seq.shape[0], -1)
            return clf.predict_proba(scaler.transform(flat))
        proba_pos = model.predict(x_seq, verbose=0).flatten()
        return np.column_stack([1.0 - proba_pos, proba_pos])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)
