"""Continuous-state DGPs: local level, local linear trend, AR(1) state,
time-varying volatility. All accept an optional list of BreakSpec.

Reference scales (sigma_ref), used to express break magnitudes:
  - LocalLevelDGP / LocalLinearTrendDGP: total one-step observation scale
    sqrt(q + r) (the state is a random walk, so no stationary sd exists).
  - AR1StateDGP: stationary state sd sqrt(q / (1 - phi^2)).
  - TimeVaryingVolDGP: baseline observation sd.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .base import DGP, DGPSample
from .breaks import (
    BreakSpec,
    break_times,
    combined_level_path,
    obs_noise_scale_path,
    state_noise_scale_path,
)


@dataclass
class LocalLevelDGP(DGP):
    """S_t = S_{t-1} + w_t,  w ~ N(0, q);  Y_t = S_t + v_t,  v ~ N(0, r).

    snr = q / r. Optional heavy-tailed obs noise: Student-t with
    ``t_dof`` degrees of freedom, scaled to variance r.
    """

    q: float = 0.5
    r: float = 1.0
    breaks: list[BreakSpec] = field(default_factory=list)
    t_dof: int | None = None  # None -> Gaussian obs noise
    name: str = "local_level"

    @property
    def sigma_ref(self) -> float:
        return float(np.sqrt(self.q + self.r))

    def _obs_noise(self, rng: np.random.Generator, T: int) -> np.ndarray:
        if self.t_dof is None:
            return rng.normal(0.0, np.sqrt(self.r), T)
        nu = self.t_dof
        if nu <= 2:
            raise ValueError("t_dof must be > 2 for finite variance")
        raw = rng.standard_t(nu, T)
        return raw * np.sqrt(self.r * (nu - 2) / nu)

    def sample(self, T: int, seed: int) -> DGPSample:
        rng = self.rng(seed)
        w = rng.normal(0.0, np.sqrt(self.q), T) * state_noise_scale_path(T, self.breaks)
        v = self._obs_noise(rng, T)
        S = np.cumsum(w)
        S += combined_level_path(T, self.breaks, self.sigma_ref)
        v *= obs_noise_scale_path(T, self.breaks)
        Y = S + v
        return DGPSample(Y=Y, S_true=S, break_times=break_times(T, self.breaks))

    def null_version(self) -> "LocalLevelDGP":
        return replace(self, breaks=[])


@dataclass
class LocalLinearTrendDGP(DGP):
    """Level + slope random-walk state; observation of the level."""

    q_level: float = 0.1
    q_slope: float = 0.01
    r: float = 1.0
    breaks: list[BreakSpec] = field(default_factory=list)
    name: str = "local_linear_trend"

    @property
    def sigma_ref(self) -> float:
        return float(np.sqrt(self.q_level + self.r))

    def sample(self, T: int, seed: int) -> DGPSample:
        rng = self.rng(seed)
        slope = np.cumsum(rng.normal(0.0, np.sqrt(self.q_slope), T))
        level = np.cumsum(slope + rng.normal(0.0, np.sqrt(self.q_level), T))
        level += combined_level_path(T, self.breaks, self.sigma_ref)
        v = rng.normal(0.0, np.sqrt(self.r), T) * obs_noise_scale_path(T, self.breaks)
        Y = level + v
        return DGPSample(Y=Y, S_true=level, break_times=break_times(T, self.breaks))

    def null_version(self) -> "LocalLinearTrendDGP":
        return replace(self, breaks=[])


@dataclass
class AR1StateDGP(DGP):
    """S_t = phi * S_{t-1} + w_t;  Y_t = S_t + v_t.

    Optional nonlinear state drift (misspecification flag, SPEC §6):
    drift term ``drift_coef * tanh(S_{t-1})`` added to the transition.
    """

    phi: float = 0.95
    q: float = 0.5
    r: float = 1.0
    breaks: list[BreakSpec] = field(default_factory=list)
    drift_coef: float = 0.0
    t_dof: int | None = None
    name: str = "ar1_state"

    @property
    def sigma_ref(self) -> float:
        return float(np.sqrt(self.q / (1.0 - self.phi**2)))

    def _phi_q_paths(self, T: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-step (phi, q). A persistence break switches phi to
        ``new_phi`` and rescales q to keep the stationary state variance
        (sigma_ref^2) unchanged — a pure dynamics change."""
        phi = np.full(T, self.phi)
        q = np.full(T, self.q)
        var_stat = self.q / (1.0 - self.phi**2)
        for spec in self.breaks:
            if spec.kind == "persistence":
                t0 = spec.time(T)
                phi[t0:] = spec.new_phi
                q[t0:] = var_stat * (1.0 - spec.new_phi**2)
        return phi, q

    def sample(self, T: int, seed: int) -> DGPSample:
        rng = self.rng(seed)
        w = rng.normal(0.0, 1.0, T)
        if self.t_dof is None:
            v = rng.normal(0.0, np.sqrt(self.r), T)
        else:
            nu = self.t_dof
            v = rng.standard_t(nu, T) * np.sqrt(self.r * (nu - 2) / nu)
        shift = combined_level_path(T, self.breaks, self.sigma_ref)
        phi_t, q_t = self._phi_q_paths(T)
        q_scale = state_noise_scale_path(T, self.breaks)  # SD multiplier (q-break)
        S = np.empty(T)
        prev = rng.normal(0.0, self.sigma_ref)  # start at stationarity
        for t in range(T):
            prev = (phi_t[t] * prev + self.drift_coef * np.tanh(prev)
                    + np.sqrt(q_t[t]) * q_scale[t] * w[t])
            S[t] = prev + shift[t]
        v *= obs_noise_scale_path(T, self.breaks)
        Y = S + v
        return DGPSample(Y=Y, S_true=S, break_times=break_times(T, self.breaks))

    def null_version(self) -> "AR1StateDGP":
        return replace(self, breaks=[])


@dataclass
class TimeVaryingVolDGP(DGP):
    """Constant-mean observations whose noise std follows the break path.

    The 'latent state' here is the log observation std. Only variance
    breaks are meaningful for this DGP.
    """

    mu: float = 0.0
    r: float = 1.0
    breaks: list[BreakSpec] = field(default_factory=list)
    name: str = "time_varying_vol"

    @property
    def sigma_ref(self) -> float:
        return float(np.sqrt(self.r))

    def sample(self, T: int, seed: int) -> DGPSample:
        rng = self.rng(seed)
        scale = np.sqrt(self.r) * obs_noise_scale_path(T, self.breaks)
        Y = self.mu + rng.normal(0.0, 1.0, T) * scale
        return DGPSample(Y=Y, S_true=np.log(scale),
                         break_times=break_times(T, self.breaks))

    def null_version(self) -> "TimeVaryingVolDGP":
        return replace(self, breaks=[])
