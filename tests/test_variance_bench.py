"""Variance-benchmark (whitening ladder) tests — SPEC addendum §2.

Training-freeze: perturbing monitoring-period data must not change
ȳ_train, σ̂_train, or the ARIMA order/parameters. Statistic identity:
the raw rung must be the max of exactly the latent variance-feature
CUSUMs applied to standardized raw data (no drift between ladder
rungs). Parity: both new detectors calibrate through the shared
harness (draw-identity across methods is proven detector-agnostically
in test_calibration.test_benchmark_parity_harness).
"""
import numpy as np
import pytest

from lsc.benchmarks.arima import fit_arima_prefix
from lsc.benchmarks.variance import (
    raw_var_cusum_score,
    training_moments,
    variance_cusum_arms,
)
from lsc.dgp import AR1StateDGP
from lsc.diagnostics.alarms import calibrate
from lsc.eval.detectors import (
    make_arima_var_cusum_detector,
    make_raw_var_cusum_detector,
)

T, N_TRAIN = 300, 75
NULL = AR1StateDGP(phi=0.95, q=0.04875, r=1.0)


@pytest.fixture(scope="module")
def Y():
    return NULL.sample(T, seed=555).Y


def monitoring_perturbed(Y):
    Z = Y.copy()
    Z[N_TRAIN:] += np.linspace(2.0, -4.0, len(Y) - N_TRAIN)
    return Z


def test_raw_var_training_freeze(Y):
    a = training_moments(Y, N_TRAIN)
    b = training_moments(monitoring_perturbed(Y), N_TRAIN)
    assert a == b


def test_arima_var_training_freeze(Y):
    order_a, params_a = fit_arima_prefix(Y, N_TRAIN)
    order_b, params_b = fit_arima_prefix(monitoring_perturbed(Y), N_TRAIN)
    assert order_a == order_b
    np.testing.assert_array_equal(params_a, params_b)


def test_raw_var_mirrors_latent_statistic(Y):
    """The raw rung is EXACTLY the latent variance-feature CUSUMs
    (k = 0.25/0.05 up, k = 0.05 down) on training-standardized Y."""
    mu, sd = training_moments(Y, N_TRAIN)
    arms = variance_cusum_arms((Y - mu) / sd)
    assert set(arms) == {"up_fast", "up_slow", "down"}
    expect = np.fmax(np.fmax(arms["up_fast"], arms["up_slow"]), arms["down"])
    expect[:N_TRAIN] = np.nan
    np.testing.assert_array_equal(raw_var_cusum_score(Y, N_TRAIN), expect)


@pytest.mark.parametrize("factory", [make_raw_var_cusum_detector,
                                     make_arima_var_cusum_detector])
def test_score_masked_on_training_prefix(factory, Y):
    s = factory(N_TRAIN)(Y)
    assert np.isnan(s[:N_TRAIN]).all()
    assert np.isfinite(s[N_TRAIN:]).all()
    assert (s[N_TRAIN:] >= 0).all()


@pytest.mark.parametrize("name,factory", [
    ("raw_var_cusum", make_raw_var_cusum_detector),
    ("arima_var_cusum", make_arima_var_cusum_detector),
])
def test_parity_harness_inclusion(name, factory):
    """Both new detectors calibrate through the SAME routine, seed
    block, and budget as every other method and yield a usable
    threshold."""
    det = calibrate(name, factory(N_TRAIN), NULL, T, n_reps=25,
                    far=0.05, seed0=100_000)
    assert np.isfinite(det.threshold) and det.threshold > 0
