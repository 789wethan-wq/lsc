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
        'variance' — observation-noise (r) std multiplied by ``vol_mult``
                     from the break on (a white-component break: changes
                     the marginal variance of Y but not its
                     autocorrelation structure)
        'state_var' — state-innovation (q) std multiplied by ``vol_mult``
                     from the break on; supported by AR1StateDGP and
                     LocalLevelDGP. Scales the *SD* of the state shock by
                     the SAME ``vol_mult`` the 'variance' kind scales the
                     obs-noise SD, so "×1.5" means SD×1.5 in BOTH
                     channels (the only convention under which the
                     two-channel r-vs-q comparison is meaningful; see
                     CHANGELOG M0 2026-07-13). Unlike an r-break, a
                     q-break changes both the marginal variance AND the
                     autocorrelation structure of Y (the ARMA(1,1) MA
                     parameter θ shifts), because the state's own
                     variance is what carries the persistence.
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
    vol_mult:  multiplicative factor on the noise std — observation noise
               (variance kind) or state-innovation noise (state_var kind)
    half_life: ramp half-life in observations (ramp kind)
    """

    kind: str
    time_frac: float
    magnitude: float = 1.0
    vol_mult: float = 3.0
    half_life: int = 25
    new_phi: float = 0.995

    def __post_init__(self) -> None:
        if self.kind not in ("level", "variance", "state_var", "ramp",
                             "persistence"):
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


def state_noise_scale_path(T: int, specs: list[BreakSpec]) -> np.ndarray:
    """Multiplicative state-innovation std path from state_var (q)
    breaks — the exact structural mirror of ``obs_noise_scale_path`` for
    the state shock, so the two channels use one SD convention."""
    scale = np.ones(T)
    for spec in specs:
        if spec.kind == "state_var":
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
