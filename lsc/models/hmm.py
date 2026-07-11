"""Gaussian HMM: EM fit on the training prefix (hmmlearn), then a custom
strictly-forward filter for causal state probabilities and innovations.

hmmlearn's ``predict_proba`` is forward-backward (uses future data), so
the causal path is computed here by hand with the fitted parameters.
"""
from __future__ import annotations

import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM

from .base import Model, StateEstimate

_LOG2PI = np.log(2.0 * np.pi)


class HMMModel(Model):
    def __init__(self, n_regimes: int = 2, n_em_starts: int = 5, em_seed: int = 0):
        self.n_regimes = n_regimes
        self.n_em_starts = n_em_starts
        self.em_seed = em_seed
        self.name = f"hmm_{n_regimes}"
        self.mu: np.ndarray | None = None
        self.sig2: np.ndarray | None = None
        self.A: np.ndarray | None = None
        self.pi: np.ndarray | None = None

    def _persistent_init(self, X: np.ndarray) -> GaussianHMM:
        """Extra EM start: means at spread quantiles, persistent transmat.

        hmmlearn's default initialization uses a uniform transition
        matrix, which is a poor basin for highly persistent chains and
        frequently traps EM in local optima that badly underestimate
        persistence (observed: p00 ~0.72 vs truth 0.97). Seeding one
        start with a diag-0.95 transmat fixes this.
        """
        K = self.n_regimes
        hmm = GaussianHMM(n_components=K, covariance_type="diag",
                          n_iter=500, min_covar=1e-4,
                          init_params="", params="stmc")
        qs = np.quantile(X.ravel(), np.linspace(0.1, 0.9, K))
        hmm.means_ = qs.reshape(-1, 1)
        hmm.covars_ = np.full((K, 1), max(X.var() / K, 1e-3))
        hmm.transmat_ = np.full((K, K), 0.05 / max(K - 1, 1))
        np.fill_diagonal(hmm.transmat_, 0.95)
        hmm.startprob_ = np.full(K, 1.0 / K)
        return hmm

    def fit(self, Y_train: np.ndarray) -> "HMMModel":
        X = np.asarray(Y_train, dtype=float).reshape(-1, 1)
        starts = [self._persistent_init(X)] + [
            GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="diag",
                n_iter=500,
                random_state=self.em_seed + i,
                min_covar=1e-4,
            )
            for i in range(self.n_em_starts - 1)
        ]
        best, best_ll = None, -np.inf
        for hmm in starts:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    hmm.fit(X)
                    ll = hmm.score(X)
                except Exception:
                    continue
            if ll > best_ll:
                best, best_ll = hmm, ll
        if best is None:
            raise RuntimeError("HMM EM failed on all starts")
        order = np.argsort(best.means_.ravel())  # stable regime labels
        self.mu = best.means_.ravel()[order]
        self.sig2 = best.covars_.reshape(self.n_regimes, -1)[:, 0][order]
        self.A = best.transmat_[np.ix_(order, order)]
        self.pi = best.startprob_[order]
        # guard against degenerate rows from EM on short prefixes
        self.A = np.clip(self.A, 1e-8, None)
        self.A /= self.A.sum(axis=1, keepdims=True)
        self.pi = np.clip(self.pi, 1e-8, None)
        self.pi /= self.pi.sum()
        return self

    def _log_obs(self, y: float) -> np.ndarray:
        return -0.5 * (_LOG2PI + np.log(self.sig2) + (y - self.mu) ** 2 / self.sig2)

    def filter(self, Y: np.ndarray, compute_smoothed: bool = False) -> StateEstimate:
        if self.mu is None:
            raise RuntimeError("call fit() before filter()")
        Y = np.asarray(Y, dtype=float)
        T, K = len(Y), self.n_regimes
        alpha = np.empty((T, K))       # filtered P(s_t | Y_0..t)
        log_b = np.empty((T, K))
        filtered = np.empty(T)
        innov = np.empty(T)
        loglik = 0.0
        prev = self.pi
        for t in range(T):
            pred = prev if t == 0 else self.A.T @ alpha[t - 1]
            # one-step-ahead predictive moments -> standardized innovation
            m = float(pred @ self.mu)
            v = float(pred @ (self.sig2 + self.mu**2) - m**2)
            innov[t] = (Y[t] - m) / np.sqrt(max(v, 1e-12))
            log_b[t] = self._log_obs(Y[t])
            w = pred * np.exp(log_b[t] - log_b[t].max())
            norm = w.sum()
            alpha[t] = w / norm
            loglik += np.log(norm) + log_b[t].max()
            filtered[t] = alpha[t] @ self.mu

        smoothed = None
        if compute_smoothed:
            beta = np.ones((T, K))
            for t in range(T - 2, -1, -1):
                b = np.exp(log_b[t + 1] - log_b[t + 1].max())
                beta[t] = self.A @ (b * beta[t + 1])
                beta[t] /= beta[t].sum()
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            smoothed = gamma @ self.mu

        return StateEstimate(
            filtered=filtered,
            innovations=innov,
            smoothed=smoothed,
            filtered_probs=alpha,
            loglik=float(loglik),
            params={"mu": self.mu.copy(), "sig2": self.sig2.copy(),
                    "A": self.A.copy(), "pi": self.pi.copy()},
        )
