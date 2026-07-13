"""Fast-or-never theory for the innovation CUSUM (exp06; derivations
and propositions in experiments/THEORY.md).

Setting: steady-state Kalman filter with KNOWN parameters for
S_t = phi S_{t-1} + w_t (var q), Y_t = S_t + v_t (var r). A level break
adds delta to the state path from t0 on. By linearity of the filter,
the standardized innovations of the broken path equal the null
innovations (iid N(0,1) in steady state) plus a deterministic mean path
mu_t that decays geometrically from delta/sqrt(F) to
mu_inf = delta (1-phi) / ((1 - phi (1-K)) sqrt(F)).
If mu_inf < k (the CUSUM drift allowance), the post-transient CUSUM has
negative drift: detection happens fast (during the transient) or,
with exponentially small probability, never.
"""
from __future__ import annotations

import numpy as np


def riccati_steady_state(phi: float, q: float, r: float
                         ) -> tuple[float, float, float]:
    """Steady-state (P, K, F) for the AR(1)+noise Kalman filter.

    P is the one-step state-prediction variance, fixed point of
    P = phi^2 P r / (P + r) + q; K = P/(P+r) the gain; F = P + r the
    innovation variance.
    """
    b = r * (1.0 - phi**2) - q
    P = 0.5 * (-b + np.sqrt(b * b + 4.0 * q * r))
    K = P / (P + r)
    return float(P), float(K), float(P + r)


def arma11_representation(phi: float, q: float, r: float
                          ) -> tuple[float, float]:
    """Exact ARMA(1,1) reduced form of the observable Y for
    S_t = phi S_{t-1} + w_t (var q), Y_t = S_t + v_t (var r).

    Differencing by the AR operator gives an MA(1):
        (1 - phi L) Y_t = w_t + v_t - phi v_{t-1} =: u_t,
    with autocovariances gamma_u(0) = q + r(1 + phi^2),
    gamma_u(1) = -phi r, and gamma_u(h) = 0 for h >= 2. Matching to
    (1 - theta L) eps_t (var sigma_eps^2) via
        (1 + theta^2) sigma_eps^2 = gamma_u(0),
        -theta sigma_eps^2       = gamma_u(1),
    the invertible root is

        theta = (m - sqrt(m^2 - 4)) / 2,   m = (q + r(1+phi^2)) / (phi r),
        sigma_eps^2 = phi r / theta.

    Two identities hold at the steady state and are the content of
    Proposition 1's connection to the reduced form (verified to machine
    precision in test_theory / exp07):
      * sigma_eps^2 == F, the Riccati innovation variance (P + r);
      * theta == rho == phi (1 - K), the innovation-mean decay rate.
    The ARMA(1,1) innovations eps_t and the steady-state Kalman
    innovations are therefore the same linear innovations of the same
    Gaussian process. Returns (theta, sigma_eps^2)."""
    m = (q + r * (1.0 + phi**2)) / (phi * r)
    theta = 0.5 * (m - np.sqrt(m * m - 4.0))
    sigma_eps2 = phi * r / theta
    return float(theta), float(sigma_eps2)


def innovation_mean_path(delta: float, phi: float, K: float, F: float,
                         n_post: int) -> np.ndarray:
    """Standardized innovation mean mu_t for t = t0 .. t0+n_post-1
    after a state level shift of delta at t0 (Proposition 1):

        mu_j = (delta - phi a_{j-1}) / sqrt(F),
        a_j  = rho a_{j-1} + K delta,   rho = phi (1 - K),  a_{-1} = 0.

    mu_0 = delta/sqrt(F); mu_j -> mu_infinity geometrically at rate rho.
    """
    rho = phi * (1.0 - K)
    mu = np.empty(n_post)
    a = 0.0
    for j in range(n_post):
        mu[j] = (delta - phi * a) / np.sqrt(F)
        a = rho * a + K * delta
    return mu


def mu_infinity(delta: float, phi: float, K: float, F: float) -> float:
    """Asymptotic standardized innovation drift after a delta level
    shift: delta (1-phi) / ((1 - phi(1-K)) sqrt(F))."""
    return float(delta * (1.0 - phi) / ((1.0 - phi * (1.0 - K))
                                        * np.sqrt(F)))


def never_detect_bound(mu_inf: float, k: float, h: float,
                       L: int) -> float:
    """Finite-horizon union/Lundberg bound (Proposition 2): if
    mu_inf < k, the post-transient one-sided CUSUM with allowance k and
    threshold h alarms within L observations with probability at most
    L exp(-2 (k - mu_inf) h) (Gaussian increments). Returns 1.0 when
    mu_inf >= k (bound vacuous: positive drift)."""
    if mu_inf >= k:
        return 1.0
    return float(min(1.0, L * np.exp(-2.0 * (k - mu_inf) * h)))


def wald_delay(shift_std: float, k: float, h: float) -> float:
    """Wald first-passage approximation for a CUSUM under sustained
    standardized drift shift_std > k: expected delay ~ h/(shift_std-k)
    (Proposition 3; raw-Y CUSUM under a permanent level shift)."""
    if shift_std <= k:
        return float("inf")
    return float(h / (shift_std - k))


def steady_state_innovations(Y: np.ndarray, phi: float, q: float,
                             r: float) -> np.ndarray:
    """Standardized innovations from the steady-state (known-parameter)
    Kalman filter — the object the theory describes. The filter state
    starts at the stationary prior mean 0."""
    _, K, F = riccati_steady_state(phi, q, r)
    e = np.empty(len(Y))
    shat = 0.0
    for t, y in enumerate(Y):
        pred = phi * shat
        e[t] = (y - pred) / np.sqrt(F)
        shat = pred + K * (y - pred)
    return e
