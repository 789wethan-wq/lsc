"""ARIMA + residual-CUSUM benchmark.

Order selected by AIC over a small grid on the training prefix; the
full series is then filtered with those fixed parameters (statespace
forward recursion — causal), and the standardized one-step-ahead
residuals feed the same two-sided Page CUSUM used by the LSC
break-pressure feature.
"""
from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from lsc.diagnostics.features import break_pressure

ORDER_GRID = [(1, 0, 0), (2, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]


def arima_cusum_score(Y: np.ndarray, n_train: int, k: float = 0.5) -> np.ndarray:
    Y = np.asarray(Y, dtype=float)
    train = Y[:n_train]
    best_order, best_aic, best_params = None, np.inf, None
    for order in ORDER_GRID:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = ARIMA(train, order=order).fit()
            except Exception:
                continue
        if res.aic < best_aic:
            best_order, best_aic, best_params = order, res.aic, np.asarray(res.params)
    if best_order is None:
        raise RuntimeError("all ARIMA fits failed on training prefix")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = ARIMA(Y, order=best_order).filter(best_params)
        innov = np.asarray(full.standardized_forecasts_error).ravel()
    score = break_pressure(innov, k=k)
    score[:n_train] = np.nan
    return score
