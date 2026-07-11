"""DGP abstract base class and sample container.

Integrity constraint (SPEC §4.3): this module must not share any
parameterization code with the estimation layer in ``lsc.models``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DGPSample:
    """One simulated series with full ground truth."""

    Y: np.ndarray                       # (T,) observations
    S_true: np.ndarray                  # (T,) latent state path
    break_times: list[int] = field(default_factory=list)  # 0-based indices
    regime_path: np.ndarray | None = None  # (T,) int labels, switching DGPs only

    def __post_init__(self) -> None:
        self.Y = np.asarray(self.Y, dtype=float)
        self.S_true = np.asarray(self.S_true, dtype=float)
        if self.Y.shape != self.S_true.shape:
            raise ValueError("Y and S_true must have the same shape")


class DGP(ABC):
    """A data-generating process.

    ``sample`` must be deterministic given (T, seed) and return complete
    ground truth (SPEC §6).
    """

    name: str = "dgp"

    @abstractmethod
    def sample(self, T: int, seed: int) -> DGPSample:
        ...

    @abstractmethod
    def null_version(self) -> "DGP":
        """Matched no-break version of this DGP, used for threshold
        calibration (SPEC §4.2). For a DGP that is already null, returns
        an equivalent DGP."""
        ...

    def rng(self, seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)
