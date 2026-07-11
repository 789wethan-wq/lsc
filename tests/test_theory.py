"""Theory-layer checks (exp06): Riccati fixed point, mean-path limit,
and a fast Monte Carlo confirmation of the innovation mean path."""
import numpy as np

from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.theory import (
    innovation_mean_path,
    mu_infinity,
    never_detect_bound,
    riccati_steady_state,
    steady_state_innovations,
    wald_delay,
)

PHI, Q, R = 0.95, 0.04875, 1.0  # spec-SNR 0.5 arena


def test_riccati_fixed_point():
    P, K, F = riccati_steady_state(PHI, Q, R)
    assert np.isclose(P, PHI**2 * P * R / (P + R) + Q)
    assert np.isclose(K, P / (P + R)) and np.isclose(F, P + R)
    # the values quoted in the CHANGELOG pre-registration
    assert np.isclose(K, 0.165, atol=5e-3)
    assert np.isclose(np.sqrt(F), 1.094, atol=5e-3)


def test_mean_path_limit_and_start():
    _, K, F = riccati_steady_state(PHI, Q, R)
    delta = 3.0 * np.sqrt(Q / (1 - PHI**2))
    mu = innovation_mean_path(delta, PHI, K, F, 400)
    assert np.isclose(mu[0], delta / np.sqrt(F))
    assert np.isclose(mu[-1], mu_infinity(delta, PHI, K, F), atol=1e-10)
    assert np.all(np.diff(mu) <= 0)          # monotone geometric decay
    assert np.all(np.diff(mu[:50]) < 0)      # strictly, pre-convergence
    # knife-edge value from the pre-registration: mu_inf(3 sigma) = 0.469
    assert np.isclose(mu_infinity(delta, PHI, K, F), 0.469, atol=5e-3)


def test_innovation_mean_path_matches_filter_mc():
    """Average standardized innovations from the steady-state filter on
    broken paths must track the deterministic mean path."""
    delta = 3.0 * np.sqrt(Q / (1 - PHI**2))
    dgp = AR1StateDGP(phi=PHI, q=Q, r=R, breaks=[
        BreakSpec(kind="level", time_frac=0.5, magnitude=3.0)])
    T, t0, n_reps = 300, 150, 300
    acc = np.zeros(T - t0)
    for i in range(n_reps):
        Y = dgp.sample(T, seed=777_000 + i).Y
        acc += steady_state_innovations(Y, PHI, Q, R)[t0:]
    emp = acc / n_reps
    _, K, F = riccati_steady_state(PHI, Q, R)
    mu = innovation_mean_path(delta, PHI, K, F, T - t0)
    # MC se per time point ~ 1/sqrt(300) ~ 0.058
    assert np.max(np.abs(emp - mu)) < 4 * 0.058
    assert abs(emp[-50:].mean() - mu[-50:].mean()) < 0.03


def test_bound_and_wald_shapes():
    assert never_detect_bound(0.6, 0.5, 20.0, 250) == 1.0
    b = never_detect_bound(0.156, 0.5, 22.0, 250)
    assert b < 1e-3  # the "never" regime at 1 sigma
    assert never_detect_bound(0.469, 0.5, 22.0, 250) == 1.0  # vacuous
    assert wald_delay(1.732, 0.5, 103.0) < 90
    assert wald_delay(0.4, 0.5, 100.0) == float("inf")
