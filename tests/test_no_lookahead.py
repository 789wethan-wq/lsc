"""SPEC §9 test_no_lookahead: perturb Y[t+1:]; filtered estimates,
innovations, every diagnostic feature, and every detector score at times
<= t must be bit-identical. Run for every model and every feature.

Perturbation points are chosen after the training prefix — parameters
are fitted on Y[:n_train] by design, so causality is claimed for
t >= n_train only.
"""
import numpy as np
import pytest

from lsc.dgp import LocalLevelDGP, MarkovSwitchingDGP
from lsc.diagnostics.features import compute_features
from lsc.eval.detectors import (
    make_arima_cusum_detector,
    make_arima_var_cusum_detector,
    make_composite_detector,
    make_innovation_cusum_detector,
    make_plain_hmm_detector,
    make_raw_cusum_detector,
    make_raw_var_cusum_detector,
    make_state_cusum_detector,
)
from lsc.models import HMMModel, KalmanModel, SwitchingModel

T, N_TRAIN = 400, 120
T_CHECK = [150, 250, 380]  # perturbation points, all >= N_TRAIN


@pytest.fixture(scope="module")
def ll_Y():
    return LocalLevelDGP(q=0.5, r=1.0).sample(T, seed=777).Y


@pytest.fixture(scope="module")
def ms_Y():
    return MarkovSwitchingDGP(means=(0.0, 3.0), sigmas=(1.0, 1.0),
                              persistence=0.97).sample(T, seed=778).Y


def perturbed(Y, t):
    Z = Y.copy()
    Z[t + 1:] += np.linspace(5.0, -3.0, len(Y) - t - 1)  # arbitrary corruption
    return Z


MODEL_FACTORIES = {
    "kalman": lambda: KalmanModel(),
    "hmm": lambda: HMMModel(2),
    "switching": lambda: SwitchingModel(2),
}


@pytest.mark.parametrize("model_name", list(MODEL_FACTORIES))
@pytest.mark.parametrize("t", T_CHECK)
def test_filtered_and_innovations_causal(model_name, t, ll_Y, ms_Y):
    Y = ms_Y if model_name in ("hmm", "switching") else ll_Y
    factory = MODEL_FACTORIES[model_name]
    a = factory().fit_filter(Y, n_train=N_TRAIN)
    b = factory().fit_filter(perturbed(Y, t), n_train=N_TRAIN)
    np.testing.assert_array_equal(a.filtered[: t + 1], b.filtered[: t + 1])
    np.testing.assert_array_equal(a.innovations[: t + 1], b.innovations[: t + 1])
    if a.filtered_probs is not None:
        np.testing.assert_array_equal(a.filtered_probs[: t + 1],
                                      b.filtered_probs[: t + 1])


@pytest.mark.parametrize("model_name", ["kalman", "hmm"])
@pytest.mark.parametrize("t", T_CHECK)
def test_every_feature_causal(model_name, t, ll_Y, ms_Y):
    Y = ms_Y if model_name == "hmm" else ll_Y
    factory = MODEL_FACTORIES[model_name]
    fa = compute_features(factory().fit_filter(Y, n_train=N_TRAIN), n_train=N_TRAIN)
    fb = compute_features(factory().fit_filter(perturbed(Y, t), n_train=N_TRAIN),
                          n_train=N_TRAIN)
    from lsc.diagnostics.features import FEATURE_FNS
    assert set(fa) == set(fb) and len(fa) == len(FEATURE_FNS)
    for name in fa:
        np.testing.assert_array_equal(fa[name][: t + 1], fb[name][: t + 1],
                                      err_msg=f"feature {name} leaks lookahead")


@pytest.mark.parametrize("t", T_CHECK)
def test_detector_scores_causal(t, ll_Y, ms_Y):
    null_ll = LocalLevelDGP(q=0.5, r=1.0)
    detectors = {
        "lsc_kalman_cusum": (make_innovation_cusum_detector(lambda: KalmanModel(), N_TRAIN), ll_Y),
        "lsc_composite": (make_composite_detector(lambda: KalmanModel(), null_ll, T,
                                                  N_TRAIN, n_scale_reps=5), ll_Y),
        "lsc_state_cusum": (make_state_cusum_detector(lambda: KalmanModel(), N_TRAIN), ll_Y),
        "raw_cusum": (make_raw_cusum_detector(N_TRAIN), ll_Y),
        "arima_cusum": (make_arima_cusum_detector(N_TRAIN), ll_Y),
        "raw_var_cusum": (make_raw_var_cusum_detector(N_TRAIN), ll_Y),
        "arima_var_cusum": (make_arima_var_cusum_detector(N_TRAIN), ll_Y),
        "plain_hmm": (make_plain_hmm_detector(N_TRAIN), ms_Y),
    }
    for name, (fn, Y) in detectors.items():
        sa, sb = fn(Y), fn(perturbed(Y, t))
        np.testing.assert_array_equal(sa[: t + 1], sb[: t + 1],
                                      err_msg=f"detector {name} leaks lookahead")


def test_smoothed_is_not_causal(ll_Y):
    """Sanity check on the test itself: the smoother SHOULD change when
    the future changes — if it didn't, the perturbation would be broken."""
    t = 250
    a = KalmanModel().fit_filter(ll_Y, n_train=N_TRAIN, compute_smoothed=True)
    b = KalmanModel().fit_filter(perturbed(ll_Y, t), n_train=N_TRAIN,
                                 compute_smoothed=True)
    assert not np.array_equal(a.smoothed[: t + 1], b.smoothed[: t + 1])
