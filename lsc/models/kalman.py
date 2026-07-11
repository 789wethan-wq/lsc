"""Linear-Gaussian state-space models via statsmodels UnobservedComponents.

Parameters are MLE-fitted on the training prefix; filtering on the full
series then uses those fixed parameters, so filtered quantities are
strictly causal (statsmodels' Kalman filter is a forward recursion).
"""
from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents

from .base import Model, StateEstimate


class KalmanModel(Model):
    """Linear-Gaussian SSM. spec:
    'llevel'  — random-walk level + noise
    'lltrend' — local linear trend + noise
    'ar1'     — stationary AR(1) state + observation noise
    """

    def __init__(self, spec: str = "llevel"):
        if spec not in ("llevel", "lltrend", "ar1"):
            raise ValueError("spec must be 'llevel', 'lltrend' or 'ar1'")
        self.spec = spec
        self.name = f"kalman_{self.spec}"
        self._params: np.ndarray | None = None

    def _make(self, Y: np.ndarray) -> UnobservedComponents:
        if self.spec == "ar1":
            return UnobservedComponents(Y, level=False, irregular=True,
                                        autoregressive=1)
        return UnobservedComponents(Y, level=self.spec)

    def fit(self, Y_train: np.ndarray) -> "KalmanModel":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._make(np.asarray(Y_train, float)).fit(disp=False)
        self._params = np.asarray(res.params)
        self._param_names = list(res.param_names)
        return self

    def filter(self, Y: np.ndarray, compute_smoothed: bool = False) -> StateEstimate:
        if self._params is None:
            raise RuntimeError("call fit() before filter()")
        Y = np.asarray(Y, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = self._make(Y)
            res = mod.filter(self._params)
            smoothed = None
            if compute_smoothed:
                sres = mod.smooth(self._params)
                smoothed = np.asarray(sres.smoothed_state[0])
        filtered = np.asarray(res.filtered_state[0])
        innov = np.asarray(res.standardized_forecasts_error[0])
        return StateEstimate(
            filtered=filtered,
            innovations=innov,
            smoothed=smoothed,
            loglik=float(res.llf),
            params=dict(zip(self._param_names, self._params)),
        )
