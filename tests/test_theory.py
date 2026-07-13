"""Theory-layer checks (exp06): Riccati fixed point, mean-path limit,
and a fast Monte Carlo confirmation of the innovation mean path."""
import numpy as np

from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.theory import (
    arma11_representation,
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


def test_arma11_riccati_identities():
    """The ARMA(1,1) reduced form of Y matches the Riccati/Proposition-1
    quantities to machine precision (M1): sigma_eps^2 == F and
    theta == rho == phi(1-K), for every (phi, q, r)."""
    for phi, q, r in [(0.95, 0.04875, 1.0), (0.95, 0.00975, 1.0),
                      (0.95, 0.195, 1.0), (0.5, 0.1, 1.0),
                      (0.8, 0.1, 1.0), (0.99, 0.1, 1.0)]:
        _, K, F = riccati_steady_state(phi, q, r)
        theta, sigma2 = arma11_representation(phi, q, r)
        assert abs(sigma2 - F) < 1e-12
        assert abs(theta - phi * (1.0 - K)) < 1e-12
        assert 0.0 < theta < 1.0  # invertible root


def test_arma_kalman_equivalence():
    """M1 gate regression guard: with TRUE parameters the ARMA(1,1)
    innovations (statsmodels ARMA filter) and the steady-state Kalman
    innovations are the same series (rho_bar >= 0.95, in fact ~1). The
    two are entirely independent code paths."""
    import warnings

    from statsmodels.tsa.arima.model import ARIMA

    m0 = 125
    for phi, q, r in [(0.95, 0.00975, 1.0), (0.95, 0.04875, 1.0),
                      (0.95, 0.195, 1.0)]:
        theta, sigma2 = arma11_representation(phi, q, r)
        dgp = AR1StateDGP(phi=phi, q=q, r=r)
        rhos, deltas = [], []
        for i in range(30):
            Y = dgp.sample(500, seed=100_000 + i).Y
            e_kal = steady_state_innovations(Y, phi, q, r)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(Y, order=(1, 0, 1), trend="n").filter(
                    [phi, -theta, sigma2])
            e_ari = np.asarray(res.standardized_forecasts_error).ravel()
            rhos.append(np.corrcoef(e_kal[m0:], e_ari[m0:])[0, 1])
            deltas.append(np.max(np.abs(e_kal[m0:] - e_ari[m0:])))
        assert np.median(rhos) >= 0.95      # gate threshold (A1)
        assert np.median(rhos) > 0.999      # actually near-exact
        assert np.max(deltas) < 1e-5        # machine precision up to startup


def test_bound_and_wald_shapes():
    assert never_detect_bound(0.6, 0.5, 20.0, 250) == 1.0
    b = never_detect_bound(0.156, 0.5, 22.0, 250)
    assert b < 1e-3  # the "never" regime at 1 sigma
    assert never_detect_bound(0.469, 0.5, 22.0, 250) == 1.0  # vacuous
    assert wald_delay(1.732, 0.5, 103.0) < 90
    assert wald_delay(0.4, 0.5, 100.0) == float("inf")
