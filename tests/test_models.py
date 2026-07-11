"""M2 tests: interface contract, causal filtering, state recovery."""
import numpy as np
import pytest

from lsc.dgp import LocalLevelDGP, MarkovSwitchingDGP
from lsc.models import HMMModel, KalmanModel, SwitchingModel


@pytest.fixture(scope="module")
def ll_sample():
    return LocalLevelDGP(q=0.5, r=1.0).sample(500, seed=101)


@pytest.fixture(scope="module")
def ms_sample():
    return MarkovSwitchingDGP(means=(0.0, 3.0), sigmas=(1.0, 1.0),
                              persistence=0.97).sample(600, seed=102)


def test_kalman_state_recovery(ll_sample):
    est = KalmanModel().fit_filter(ll_sample.Y, n_train=150, compute_smoothed=True)
    assert np.corrcoef(est.filtered, ll_sample.S_true)[0, 1] > 0.95
    # smoothed should be at least as accurate as filtered
    rmse_f = np.sqrt(np.mean((est.filtered - ll_sample.S_true) ** 2))
    rmse_s = np.sqrt(np.mean((est.smoothed - ll_sample.S_true) ** 2))
    assert rmse_s <= rmse_f


def test_kalman_innovations_white_under_null(ll_sample):
    est = KalmanModel().fit_filter(ll_sample.Y, n_train=150)
    e = est.innovations[20:]  # drop diffuse warmup
    assert abs(e.mean()) < 0.2
    assert 0.8 < e.std() < 1.25
    ac1 = np.corrcoef(e[:-1], e[1:])[0, 1]
    assert abs(ac1) < 0.15


def test_hmm_state_recovery(ms_sample):
    est = HMMModel(2).fit_filter(ms_sample.Y, n_train=300, compute_smoothed=True)
    assert np.corrcoef(est.filtered, ms_sample.S_true)[0, 1] > 0.9
    assert est.filtered_probs.shape == (600, 2)
    np.testing.assert_allclose(est.filtered_probs.sum(axis=1), 1.0, atol=1e-9)


def test_switching_state_recovery(ms_sample):
    est = SwitchingModel(2).fit_filter(ms_sample.Y, n_train=300)
    assert np.corrcoef(est.filtered, ms_sample.S_true)[0, 1] > 0.9


@pytest.mark.parametrize("model_factory", [
    lambda: KalmanModel(),
    lambda: HMMModel(2),
], ids=["kalman", "hmm"])
def test_filter_requires_fit(model_factory):
    with pytest.raises(RuntimeError):
        model_factory().filter(np.zeros(50))


@pytest.mark.parametrize("model_factory", [
    lambda: KalmanModel(),
    lambda: HMMModel(2),
], ids=["kalman", "hmm"])
def test_filter_is_deterministic(model_factory, ll_sample):
    m = model_factory()
    m.fit(ll_sample.Y[:150])
    a = m.filter(ll_sample.Y)
    b = m.filter(ll_sample.Y)
    np.testing.assert_array_equal(a.filtered, b.filtered)
    np.testing.assert_array_equal(a.innovations, b.innovations)
