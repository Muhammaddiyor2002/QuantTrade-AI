"""Portfolio-level risk analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def var_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical Value-at-Risk at confidence ``1 - alpha``."""
    if returns.empty:
        return 0.0
    return float(-np.quantile(returns.dropna(), alpha))


def cvar_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical Conditional Value-at-Risk (Expected Shortfall)."""
    if returns.empty:
        return 0.0
    cutoff = np.quantile(returns.dropna(), alpha)
    tail = returns[returns <= cutoff]
    if tail.empty:
        return 0.0
    return float(-tail.mean())


def kelly_fraction(mean_return: float, var_return: float) -> float:
    """Continuous Kelly fraction; clipped to [0, 1]."""
    if var_return <= 0:
        return 0.0
    return float(max(0.0, min(1.0, mean_return / var_return)))


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation of returns columns."""
    return returns.corr().fillna(0.0)


def portfolio_volatility(weights: np.ndarray, cov: pd.DataFrame) -> float:
    weights = np.asarray(weights, dtype=float)
    cov_arr = cov.to_numpy()
    return float(np.sqrt(weights @ cov_arr @ weights))


def equal_weight_portfolio(returns: pd.DataFrame) -> pd.Series:
    n = returns.shape[1]
    if n == 0:
        return pd.Series(dtype=float)
    return pd.Series(np.ones(n) / n, index=returns.columns, name="weight")


def risk_parity_weights(returns: pd.DataFrame) -> pd.Series:
    """Naive risk-parity: weight inversely proportional to per-asset volatility."""
    vol = returns.std(ddof=0)
    if (vol == 0).all():
        return equal_weight_portfolio(returns)
    inv = 1.0 / vol.replace(0.0, np.nan)
    weights = (inv / inv.sum()).fillna(0.0)
    weights.name = "weight"
    return weights
