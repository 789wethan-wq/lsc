"""Common estimation interface (SPEC §5, lsc/models/base.py).

Causality contract (SPEC §4.1): ``fit`` may only ever see a training
prefix of the series — full-sample MLE would leak future data into
"online" detection claims. ``filter`` then runs a forward-only
recursion with the fitted parameters held fixed, so ``filtered[t]`` and
``innovations[t]`` depend on Y[0..t] only. ``smoothed`` uses the full
sample and must only be reported as retrospective/offline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StateEstimate:
    filtered: np.ndarray            # (T,) E[S_t | Y_0..t], causal
    innovations: np.ndarray         # (T,) standardized 1-step-ahead errors, causal
    smoothed: np.ndarray | None = None  # (T,) E[S_t | Y_0..T-1], offline only
    filtered_probs: np.ndarray | None = None  # (T, K) regime probs, discrete models
    loglik: float = np.nan
    params: dict = field(default_factory=dict)


class Model(ABC):
    """fit-on-prefix / filter-forward estimation model."""

    name: str = "model"

    @abstractmethod
    def fit(self, Y_train: np.ndarray) -> "Model":
        """Estimate parameters from a training prefix only. Returns self."""
        ...

    @abstractmethod
    def filter(self, Y: np.ndarray, compute_smoothed: bool = False) -> StateEstimate:
        """Causal forward filtering with fixed parameters."""
        ...

    def fit_filter(self, Y: np.ndarray, n_train: int,
                   compute_smoothed: bool = False) -> StateEstimate:
        self.fit(np.asarray(Y, dtype=float)[:n_train])
        return self.filter(np.asarray(Y, dtype=float),
                           compute_smoothed=compute_smoothed)
