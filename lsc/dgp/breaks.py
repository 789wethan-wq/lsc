"""Structural-break specifications and injection utilities.

Breaks are specified in T-independent units: times as fractions of the
sample, magnitudes in multiples of a DGP-specific reference scale
``sigma_ref`` (documented per DGP).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BreakSpec:
    """One structural break.

    kind:
        'level'    — additive shift of the latent state from the break on
        'variance' — observation-noise std multiplied by ``vol_mult`` from
                     the break on
        'ramp'     — gradual logistic level shift centered at the break
                     time with the given half-life (obs until the shift
                     reaches half its final size)
        'persistence' — AR(1) state coefficient jumps to ``new_phi``;
                     supported by AR1StateDGP only, which rescales the
                     state-noise variance to keep the stationary state
                     variance unchanged (a pure dynamics change:
                     marginal mean and variance of Y are preserved)
    time_frac: break location as a fraction of T (0 < time_frac < 1)
    magnitude: size in units of the DGP's sigma_ref (level/ramp kinds)
    vol_mult:  multiplicative factor on observation noise std (variance kind)
    half_life: ramp half-life in observations (ramp kind)
    """

    kind: str
    time_frac: float
    magnitude: float = 1.0
    vol_mult: float = 3.0
    half_life: int = 25
    new_phi: float = 0.995

    def __post_init__(self) -> None:
        if self.kind not in ("level", "variance", "ramp", "persistence"):
            raise ValueError(f"unknown break kind: {self.kind}")
        if not 0.0 < self.time_frac < 1.0:
            raise ValueError("time_frac must be in (0, 1)")

    def time(self, T: int) -> int:
        return int(round(self.time_frac * T))


def level_shift_path(T: int, spec: BreakSpec, sigma_ref: float) -> np.ndarray:
    """Additive latent-state shift path implied by a level or ramp break."""
    t0 = spec.time(T)
    delta = spec.magnitude * sigma_ref
    t = np.arange(T)
    if spec.kind == "level":
        return np.where(t >= t0, delta, 0.0)
    if spec.kind == "ramp":
        # logistic centered at t0; half_life obs from center to half of
        # the remaining rise: logistic(k*h) - logistic(0) = 0.25 -> k*h = ln(3)
        k = np.log(3.0) / max(spec.half_life, 1)
        return delta / (1.0 + np.exp(-k * (t - t0)))
    raise ValueError(f"{spec.kind} is not a level-type break")


def obs_noise_scale_path(T: int, specs: list[BreakSpec]) -> np.ndarray:
    """Multiplicative observation-noise std path from variance breaks."""
    scale = np.ones(T)
    for spec in specs:
        if spec.kind == "variance":
            scale[spec.time(T):] *= spec.vol_mult
    return scale


def combined_level_path(T: int, specs: list[BreakSpec], sigma_ref: float) -> np.ndarray:
    """Sum of all level/ramp shift paths (supports multiple breaks)."""
    path = np.zeros(T)
    for spec in specs:
        if spec.kind in ("level", "ramp"):
            path += level_shift_path(T, spec, sigma_ref)
    return path


def break_times(T: int, specs: list[BreakSpec]) -> list[int]:
    return sorted(spec.time(T) for spec in specs)
