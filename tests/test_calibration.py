"""SPEC §9 integrity tests: test_null_far_calibration and
test_benchmark_parity_harness.

Seeds are fixed, so these tests are deterministic: the FAR tolerance
check either always passes or always fails for a given code state.
"""
import hashlib

import numpy as np
import pytest

from lsc.dgp import AR1StateDGP
from lsc.diagnostics.alarms import calibrate, empirical_far
from lsc.eval.detectors import (
    make_innovation_cusum_detector,
    make_raw_cusum_detector,
)
from lsc.models import KalmanModel

T, N_TRAIN = 500, 125
NULL = AR1StateDGP(phi=0.95, q=0.5, r=1.0)
FAR = 0.05
TOL = 0.015  # ±1.5 pp (SPEC §9)

# Calibration budgets per detector. The 95th-percentile threshold is an
# order statistic, so its exceedance probability is Beta(n+1-k, k)
# regardless of the score distribution — with n=300 that alone has
# ~1.3 pp sd, too loose for the ±1.5 pp tolerance when combined with
# check noise. raw_cusum (cheap, and with an extremely heavy-tailed null
# max on autocorrelated data) therefore gets a larger budget. The
# comparison EXPERIMENT still uses one common budget for all methods
# (SPEC §9 parity) — this test only verifies the calibration routine
# hits its target given adequate replications.
@pytest.mark.parametrize("name,factory,n_cal,n_check", [
    ("raw_cusum", lambda: make_raw_cusum_detector(N_TRAIN), 1500, 1000),
    ("lsc_kalman_cusum",
     lambda: make_innovation_cusum_detector(lambda: KalmanModel("ar1"), N_TRAIN),
     300, 300),
])
def test_null_far_calibration(name, factory, n_cal, n_check):
    det = calibrate(name, factory(), NULL, T, n_reps=n_cal, far=FAR,
                    seed0=100_000)
    far = empirical_far(det, NULL, T, n_reps=n_check, seed0=300_000)
    assert abs(far - FAR) <= TOL, (
        f"{name}: empirical FAR {far:.3f} outside {FAR}±{TOL}")


def test_benchmark_parity_harness():
    """All methods must consume identical null draws and identical
    calibration budgets: the harness derives draws only from
    (null_dgp, T, seed0, n_reps), never from the detector."""
    seen: dict[str, list[str]] = {}

    def probe(tag):
        def fn(Y):
            seen.setdefault(tag, []).append(
                hashlib.sha1(np.ascontiguousarray(Y).tobytes()).hexdigest())
            out = np.full(len(Y), np.nan)
            out[N_TRAIN:] = 0.0
            return out
        return fn

    for tag in ("method_a", "method_b"):
        calibrate(tag, probe(tag), NULL, T, n_reps=25, far=FAR, seed0=100_000)
    assert seen["method_a"] == seen["method_b"]
    assert len(seen["method_a"]) == 25
