"""Discrete-state DGPs: Gaussian Markov regime switching."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .base import DGP, DGPSample


@dataclass
class MarkovSwitchingDGP:
    """K-regime Gaussian observations, regime follows a Markov chain.

    means / sigmas: per-regime observation mean and std.
    persistence: probability of staying in the current regime; the
    remaining mass is split evenly among the other regimes.

    S_true is the active regime mean; break_times are the indices where
    the regime changes (t such that regime[t] != regime[t-1]).
    """

    means: tuple[float, ...] = (0.0, 2.0)
    sigmas: tuple[float, ...] = (1.0, 1.0)
    persistence: float = 0.98
    start_regime: int = 0
    name: str = "markov_switching"

    def __post_init__(self) -> None:
        if len(self.means) != len(self.sigmas):
            raise ValueError("means and sigmas must have equal length")
        if not 0.0 < self.persistence <= 1.0:
            raise ValueError("persistence must be in (0, 1]")

    @property
    def n_regimes(self) -> int:
        return len(self.means)

    @property
    def sigma_ref(self) -> float:
        return float(self.sigmas[0])

    def transition_matrix(self) -> np.ndarray:
        K = self.n_regimes
        if K == 1:
            return np.ones((1, 1))
        off = (1.0 - self.persistence) / (K - 1)
        P = np.full((K, K), off)
        np.fill_diagonal(P, self.persistence)
        return P

    def rng(self, seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)

    def sample(self, T: int, seed: int) -> DGPSample:
        rng = self.rng(seed)
        P = self.transition_matrix()
        regimes = np.empty(T, dtype=int)
        state = self.start_regime
        for t in range(T):
            regimes[t] = state
            state = rng.choice(self.n_regimes, p=P[state])
        means = np.asarray(self.means)[regimes]
        sigmas = np.asarray(self.sigmas)[regimes]
        Y = means + rng.normal(0.0, 1.0, T) * sigmas
        changes = list(np.nonzero(np.diff(regimes))[0] + 1)
        return DGPSample(Y=Y, S_true=means.astype(float),
                         break_times=[int(c) for c in changes],
                         regime_path=regimes)

    def null_version(self) -> "MarkovSwitchingDGP":
        """Matched null: same regime-0 marginal distribution, no switching."""
        return replace(self, persistence=1.0)


DGP.register(MarkovSwitchingDGP)
