"""ARIMA wrapped as a `Model` (fit-on-prefix / causal-filter interface).

Built so the EXISTING composite-feature machinery
(`lsc.diagnostics.features.compute_features`,
`lsc.eval.detectors.make_composite_detector`) can run unmodified on an
ARIMA-based state estimate instead of a Kalman one -- the
composite-on-ARIMA ablation (SPEC follow-up; see
`experiments/exp20_composite_on_arima.py`) asks whether the
composite's power over plain ARIMA-CUSUM comes only from richer
statistics on the same whitened innovation series (the ARMA(1,1)
equivalence proven in `lsc.theory.arma11_representation` and
`experiments/exp07_arma_equivalence.py` is a statement about that
innovation series only) or partly from something genuinely
state-specific.

ARIMA has no separate latent state the way the Kalman AR1 model's
filtered state does -- Y IS the process being modeled, not a noisy
observation of a hidden one. The model's own one-step-ahead
conditional-mean prediction of Y (`fittedvalues`, verified to sit on
the ORIGINAL Y scale even for differenced orders, e.g. (0,1,1)) is the
best available state-analog and is used as such; its standardized
one-step-ahead forecast errors are the innovations-analog, the SAME
series `lsc.benchmarks.arima.arima_standardized_residuals` already
uses for the arima_cusum / arima_var_cusum benchmarks. Both are causal
by construction of the fixed-parameter statespace forward recursion.
"""
from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from .base import Model, StateEstimate


class ARIMAModel(Model):
    """AIC-selected ARIMA (order grid in lsc.benchmarks.arima.ORDER_GRID),
    fit on the training prefix, forward-filtered with fixed parameters."""

    name = "arima"

    def __init__(self) -> None:
        self._order: tuple | None = None
        self._params: np.ndarray | None = None

    def fit(self, Y_train: np.ndarray) -> "ARIMAModel":
        from lsc.benchmarks.arima import fit_arima_prefix  # deferred: avoids a
        # circular import (lsc.benchmarks.arima -> lsc.diagnostics.features ->
        # lsc.models.base, and lsc.models/__init__ imports this module), same
        # pattern used by lsc.eval.detectors / lsc.benchmarks.variance.

        Y_train = np.asarray(Y_train, dtype=float)
        self._order, self._params = fit_arima_prefix(Y_train, len(Y_train))
        return self

    def filter(self, Y: np.ndarray, compute_smoothed: bool = False) -> StateEstimate:
        if self._order is None:
            raise RuntimeError("call fit() before filter()")
        Y = np.asarray(Y, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if compute_smoothed:
                # exp41 (SPEC R7 D): .smooth() (not .filter()) runs the
                # fixed-interval (two-sided) Kalman smoother over the
                # SAME frozen training-prefix (order, params) fit --
                # .filter() alone leaves smoother_results empty. NOT
                # causal -- conditions on the whole series, past and
                # future -- an oracle-status state proxy, the same
                # caveat already given to exp37's break-aware GARCH
                # refit and lsc.benchmarks.variance.known_*_var_cusum_score.
                res = ARIMA(Y, order=self._order).smooth(self._params)
            else:
                res = ARIMA(Y, order=self._order).filter(self._params)
        innov = np.asarray(res.standardized_forecasts_error, dtype=float).ravel()
        if compute_smoothed:
            filtered = np.asarray(res.smoother_results.smoothed_forecasts,
                                  dtype=float).ravel()
        else:
            filtered = np.asarray(res.fittedvalues, dtype=float)
        return StateEstimate(
            filtered=filtered,
            innovations=innov,
            loglik=float(res.llf),
            params=dict(order=self._order),
        )


class SmoothedARIMAModel(ARIMAModel):
    """exp41 (SPEC R7 D): identical fit to ARIMAModel, but `fit_filter`
    always requests the smoothed (two-sided) state proxy in place of
    the one-step-ahead `fittedvalues` -- so it can drop into
    `make_composite_detector`'s `model_factory` slot unchanged. NOT
    causal (see ARIMAModel.filter's compute_smoothed docstring); for
    the composite-on-ARIMA gap ablation only, not a deployable
    detector."""

    name = "smoothed_arima"

    def fit_filter(self, Y: np.ndarray, n_train: int,
                   compute_smoothed: bool = False) -> StateEstimate:
        return super().fit_filter(Y, n_train, compute_smoothed=True)
