"""AR2StateDGP tests (SPEC R2 M2): stationarity of the two chosen
parameterizations, sigma_ref against empirical stationary variance,
and the level/variance/state_var break conventions carried over
unchanged from AR1StateDGP."""
import numpy as np
import pytest

from lsc.dgp import AR2StateDGP, BreakSpec, matched_null

REAL_ROOTS = dict(phi1=1.4, phi2=-0.45)   # poles ~{0.5, 0.9}
COMPLEX_ROOTS = dict(phi1=1.6, phi2=-0.9)  # poles complex, modulus ~0.949


def _poles(phi1: float, phi2: float) -> np.ndarray:
    """Companion-matrix eigenvalues: roots of x^2 - phi1*x - phi2 = 0."""
    return np.roots([1.0, -phi1, -phi2])


@pytest.mark.parametrize("params", [REAL_ROOTS, COMPLEX_ROOTS])
def test_parameterizations_are_stationary(params):
    poles = _poles(params["phi1"], params["phi2"])
    assert np.all(np.abs(poles) < 1.0)


def test_real_roots_are_real_complex_roots_are_complex():
    real_poles = _poles(**REAL_ROOTS)
    complex_poles = _poles(**COMPLEX_ROOTS)
    assert np.all(np.isreal(real_poles))
    assert np.any(np.iscomplex(complex_poles))
    np.testing.assert_allclose(sorted(np.abs(complex_poles)),
                               [np.sqrt(0.9), np.sqrt(0.9)], atol=1e-9)


@pytest.mark.parametrize("params", [REAL_ROOTS, COMPLEX_ROOTS])
def test_sigma_ref_matches_empirical_stationary_variance(params):
    dgp = AR2StateDGP(q=0.5, r=0.0, breaks=[], burn_in=2000, **params)
    s = dgp.sample(20_000, seed=1)
    empirical_sd = s.S_true.std()
    assert 0.9 < empirical_sd / dgp.sigma_ref < 1.1


def test_level_break_shifts_state_without_feeding_back():
    T = 2000
    spec = BreakSpec("level", 0.5, magnitude=2.0)
    dgp = AR2StateDGP(q=0.5, r=1.0, breaks=[spec], **REAL_ROOTS)
    null = matched_null(dgp)
    s, s0 = dgp.sample(T, seed=3), null.sample(T, seed=3)
    delta = s.S_true - s0.S_true
    np.testing.assert_allclose(delta[:1000], 0.0, atol=1e-9)
    np.testing.assert_allclose(delta[1000:], 2.0 * dgp.sigma_ref, atol=1e-9)


def test_variance_break_scales_obs_noise():
    T = 400
    spec = BreakSpec("variance", 0.5, vol_mult=3.0)
    dgp = AR2StateDGP(q=0.0, r=1.0, breaks=[spec], **REAL_ROOTS)
    s = dgp.sample(T, seed=11)
    resid = s.Y - s.S_true
    pre, post = resid[:200].std(), resid[200:].std()
    assert 2.0 < post / pre < 4.0


def test_state_var_break_scales_shock_sd():
    T = 6000
    spec = BreakSpec("state_var", 0.5, vol_mult=1.5)
    dgp = AR2StateDGP(q=0.2, r=0.0, breaks=[spec], burn_in=500, **REAL_ROOTS)
    s = dgp.sample(T, seed=13)
    phi1, phi2 = REAL_ROOTS["phi1"], REAL_ROOTS["phi2"]
    w = s.S_true[2:] - phi1 * s.S_true[1:-1] - phi2 * s.S_true[:-2]
    pre, post = w[100:2900].std(), w[3100:].std()
    assert 1.35 < post / pre < 1.65


def test_seed_reproducibility_and_null_matched():
    dgp = AR2StateDGP(q=0.5, r=1.0, breaks=[BreakSpec("level", 0.5, magnitude=1.0)],
                      **COMPLEX_ROOTS)
    a = dgp.sample(300, seed=42)
    b = dgp.sample(300, seed=42)
    np.testing.assert_array_equal(a.Y, b.Y)
    null = dgp.null_version()
    assert null.breaks == []
