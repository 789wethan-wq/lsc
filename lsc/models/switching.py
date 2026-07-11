"""Markov-switching model (v1: statsmodels MarkovRegression, per SPEC).

Limitations (documented per SPEC §5): MarkovRegression switches the
observation mean/variance, not a full switching linear-Gaussian state —
a true switching SSM (Kim filter) is deferred to v2. Parameters are
fitted on the training prefix; the full series is then filtered with
those fixed parameters via ``.filter()``, whose
``filtered_marginal_probabilities`` are causal (Hamilton filter).
"""
from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from .base import Model, StateEstimate


class SwitchingModel(Model):
    def __init__(self, n_regimes: int = 2, switching_variance: bool = False,
                 fit_seed: int = 0):
        self.n_regimes = n_regimes
        self.switching_variance = switching_variance
        self.fit_seed = fit_seed
        self.name = f"switching_{n_regimes}"
        self._params: np.ndarray | None = None

    def _make(self, Y: np.ndarray) -> MarkovRegression:
        return MarkovRegression(
            Y, k_regimes=self.n_regimes,
            switching_variance=self.switching_variance,
        )

    def fit(self, Y_train: np.ndarray) -> "SwitchingModel":
        Y_train = np.asarray(Y_train, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # statsmodels' random start-parameter search uses the global
            # numpy RNG; seed it so fit is deterministic (required by
            # test_no_lookahead, which refits on identical prefixes)
            state = np.random.get_state()
            np.random.seed(self.fit_seed)
            try:
                res = self._make(Y_train).fit(search_reps=10)
            finally:
                np.random.set_state(state)
        self._params = np.asarray(res.params)
        self._param_names = list(res.model.param_names)
        return self

    def filter(self, Y: np.ndarray, compute_smoothed: bool = False) -> StateEstimate:
        if self._params is None:
            raise RuntimeError("call fit() before filter()")
        Y = np.asarray(Y, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = self._make(Y)
            res = mod.filter(self._params)
            probs = np.asarray(res.filtered_marginal_probabilities)  # (T, K)
            mus = np.array([
                self._params[list(self._param_names).index(f"const[{k}]")]
                for k in range(self.n_regimes)
            ])
            filtered = probs @ mus
            # standardized 1-step-ahead innovations from predicted probs
            pred = np.asarray(res.predicted_marginal_probabilities)  # (T, K)
            if self.switching_variance:
                sig2 = np.array([
                    self._params[list(self._param_names).index(f"sigma2[{k}]")]
                    for k in range(self.n_regimes)
                ])
            else:
                sig2 = np.full(self.n_regimes,
                               self._params[list(self._param_names).index("sigma2")])
            m = pred @ mus
            v = pred @ (sig2 + mus**2) - m**2
            innov = (Y - m) / np.sqrt(np.maximum(v, 1e-12))
            smoothed = None
            if compute_smoothed:
                sres = mod.smooth(self._params)
                smoothed = np.asarray(sres.smoothed_marginal_probabilities) @ mus
        return StateEstimate(
            filtered=filtered,
            innovations=innov,
            smoothed=smoothed,
            filtered_probs=probs,
            loglik=float(res.llf),
            params=dict(zip(self._param_names, self._params)),
        )
