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


def order_known_var_cusum_score(Y: np.ndarray, n_train: int,
                                order: tuple = (1, 0, 1)) -> np.ndarray:
    """Order-known ARIMA rung (exp30, SPEC R3 M1): the identical
    statistic on an ARIMA model's standardized one-step residuals, with
    the order FIXED at the true DGP order (1,0,1) instead of AIC-
    selected -- coefficients still fit by MLE on the training prefix.
    Sits between arima_var_cusum_score (order AND coefficients
    estimated) and known_kalman_var_cusum_score (order AND coefficients
    known), isolating order-selection error from coefficient noise."""
    from lsc.benchmarks.arima import arima_standardized_residuals_fixed_order

    z = arima_standardized_residuals_fixed_order(np.asarray(Y, dtype=float),
                                                 n_train, order)
    return _max_over_arms(z, n_train)


def raw_var_arm_at(Y: np.ndarray, n_train: int, t: int) -> str:
    """Which arm is maximal at time t (alarm attribution, real data)."""
    Y = np.asarray(Y, dtype=float)
    mu, sd = training_moments(Y, n_train)
    arms = variance_cusum_arms((Y - mu) / sd)
    return max(arms, key=lambda a: np.nan_to_num(arms[a][t], nan=-np.inf))


def known_raw_var_cusum_score(Y: np.ndarray, phi: float, q: float, r: float,
                              n_train: int) -> np.ndarray:
    """Known-parameter counterpart of raw_var_cusum_score (peer review
    round 3, Missing Experiments -- the Table 2 flagship level-shift
    cell's known-vs-estimated ablation, exp10, extended to the variance
    ladder): standardize by the DGP's analytic stationary mean (0) and
    SD (sqrt(q/(1-phi^2) + r), the AR(1) state's stationary variance
    plus the observation-noise variance) instead of the training-prefix
    SAMPLE mean/SD. Isolates how much of raw_var_cusum's estimation gap
    (if any) comes from using a finite-sample moment estimate rather
    than the true population one."""
    Y = np.asarray(Y, dtype=float)
    sd = float(np.sqrt(q / max(1.0 - phi**2, 1e-12) + r))
    return _max_over_arms(Y / max(sd, 1e-12), n_train)


def known_kalman_var_cusum_score(Y: np.ndarray, phi: float, q: float,
                                 r: float, n_train: int) -> np.ndarray:
    """Known-parameter counterpart of the Kalman/latent variance rung:
    the same three-arm max-CUSUM on the STEADY-STATE (true-parameter)
    Kalman innovations (lsc.theory.steady_state_innovations) instead
    of innovations from an MLE-fit KalmanModel. The natural "known"
    reference point for arima_var_cusum's estimated whitening."""
    from lsc.theory import steady_state_innovations

    e = steady_state_innovations(np.asarray(Y, dtype=float), phi, q, r)
    return _max_over_arms(e, n_train)


def est_kalman_var_cusum_score(Y: np.ndarray, n_train: int,
                               spec: str = "ar1") -> np.ndarray:
    """Estimated-parameter counterpart of known_kalman_var_cusum_score
    (SPEC_R8_missing_experiments.md exp44): the identical three-arm
    variance CUSUM statistic on the standardized one-step innovations
    of an MLE-fit KalmanModel (fit on the training prefix, then
    forward-filtered with frozen parameters), in place of
    steady_state_innovations under the TRUE parameters. Only the
    innovation source differs from known_kalman_var_cusum_score --
    same _max_over_arms call as every other rung."""
    from lsc.models import KalmanModel

    est = KalmanModel(spec).fit_filter(np.asarray(Y, dtype=float), n_train=n_train)
    return _max_over_arms(est.innovations, n_train)


def windowed_raw_var_score(Y: np.ndarray, n_train: int,
                           window: int = 60) -> np.ndarray:
    """Bounded-memory, MOSUM-style variance-changepoint statistic --
    the variance-channel counterpart of
    lsc.benchmarks.changepoint.windowed_raw_cusum_score (which fixes
    the second-event miss for MEAN shifts, exp04/CHANGELOG P2
    2026-07-16, but leaves the variance-channel second event
    undetected: "Windowed-CUSUM fix, level->var / var->var 2nd event:
    no improvement" -- because that fix's two-window statistic
    compares window MEANS, which a pure variance change does not move).

    Two sliding windows of standardized Y (frozen training-prefix
    moments, same convention as raw_var_cusum_score); statistic is a
    log-variance-ratio z-score, log(test_var/ref_var) rescaled by its
    delta-method SE (Var(log(sample_var)) ~= 2/window under the null
    for each window, so the ratio's SE ~= sqrt(4/window)) -- the
    natural windowed analog of an F-test on two variance estimates.
    Both windows slide together, so `window` obs after a permanent
    variance shift both windows sit inside the new regime and the
    statistic returns to its null level (drains), the same mechanism
    windowed_raw_cusum_score uses for mean shifts."""
    Y = np.asarray(Y, dtype=float)
    mu, sd = training_moments(Y, n_train)
    z = (Y - mu) / sd
    T = len(z)
    out = np.full(T, np.nan)
    se = 2.0 / np.sqrt(window)
    for t in range(n_train, T):
        if t - n_train - 2 * window + 1 < 0:
            continue
        test = z[t - window + 1: t + 1]
        ref = z[t - 2 * window + 1: t - window + 1]
        test_var = max(float(np.mean(test**2)), 1e-12)
        ref_var = max(float(np.mean(ref**2)), 1e-12)
        out[t] = abs(np.log(test_var / ref_var)) / se
    return out
