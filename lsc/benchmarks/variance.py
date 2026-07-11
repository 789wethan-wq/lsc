"""Raw-data and ARIMA-whitened variance-CUSUM benchmarks — the bottom
two rungs of the whitening ladder (SPEC addendum §2).

The statistic is IDENTICAL at every rung and reuses the latent
variance-feature code verbatim: up-arm Page CUSUMs of z²−1 with
allowances k = 0.25 and k = 0.05 (variance_pressure /
variance_pressure_slow), a down-arm (quieting) CUSUM of 1−z² with
k = 0.05 (variance_quiet); the detector score is the max over the
three arms, with no per-time-point standardization (standalone
detector, same treatment as lsc_tail_cusum). Only the information set
changes across rungs:

  raw    z_t = (Y_t − ȳ_train)/σ̂_train, moments frozen from the
         training prefix
  arima  standardized one-step residuals of the frozen training-prefix
         ARIMA model (whitened, not state-aware) — used directly, the
         same way the latent rung uses the Kalman filter's own
         standardized innovations
  latent (existing, lsc/diagnostics/features.py) standardized Kalman
         innovations
"""
from __future__ import annotations

import numpy as np

from lsc.diagnostics.features import variance_pressure, variance_quiet

ARM_KS = {"up_fast": 0.25, "up_slow": 0.05, "down": 0.05}


def training_moments(Y: np.ndarray, n_train: int) -> tuple[float, float]:
    """(ȳ_train, σ̂_train), frozen from the training prefix only."""
    train = np.asarray(Y, dtype=float)[:n_train]
    return float(train.mean()), float(max(train.std(ddof=1), 1e-12))


def variance_cusum_arms(z: np.ndarray) -> dict[str, np.ndarray]:
    """The three ladder arms on a standardized series z (any rung)."""
    return {
        "up_fast": variance_pressure(z, k=ARM_KS["up_fast"]),
        "up_slow": variance_pressure(z, k=ARM_KS["up_slow"]),
        "down": variance_quiet(z, k=ARM_KS["down"]),
    }


def _max_over_arms(z: np.ndarray, n_train: int) -> np.ndarray:
    arms = variance_cusum_arms(z)
    score = np.fmax(np.fmax(arms["up_fast"], arms["up_slow"]), arms["down"])
    score[:n_train] = np.nan
    return score


def raw_var_cusum_score(Y: np.ndarray, n_train: int) -> np.ndarray:
    """Bottom rung: variance CUSUM on raw Y standardized by frozen
    training-prefix moments."""
    Y = np.asarray(Y, dtype=float)
    mu, sd = training_moments(Y, n_train)
    return _max_over_arms((Y - mu) / sd, n_train)


def arima_var_cusum_score(Y: np.ndarray, n_train: int) -> np.ndarray:
    """Middle rung: the identical statistic on the frozen ARIMA model's
    standardized one-step residuals."""
    from lsc.benchmarks.arima import arima_standardized_residuals

    z = arima_standardized_residuals(np.asarray(Y, dtype=float), n_train)
    return _max_over_arms(z, n_train)


def raw_var_arm_at(Y: np.ndarray, n_train: int, t: int) -> str:
    """Which arm is maximal at time t (alarm attribution, real data)."""
    Y = np.asarray(Y, dtype=float)
    mu, sd = training_moments(Y, n_train)
    arms = variance_cusum_arms((Y - mu) / sd)
    return max(arms, key=lambda a: np.nan_to_num(arms[a][t], nan=-np.inf))
