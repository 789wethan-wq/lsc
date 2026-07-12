"""Composite-score regression guards (added 2026-07-12).

The pre-git "pooled scale -> per-time-point standardization" fix (§8.4)
changed the composite's output, but no test pinned that output, so a
stale grid_v3b parquet (robust2 = 0.058, pre-fix) survived in the repo
until the full `make all` reproduction gate regenerated it to the
correct 0.582. These tests pin the current, hand-verified behavior so a
silent composite change — or a stale artifact — is caught immediately:

  - determinism: two independent builds must agree bit-for-bit (the
    null-scale calibration is seeded; nothing may leak global RNG);
  - golden score: the max-over-features composite on a fixed break path
    must match hand-computed values (see CHANGELOG 2026-07-12);
  - feature-alive: the exceedance-indicator feature inside the composite
    is diluted, NOT dead — its standardized post-break score is large.
"""
import numpy as np
import pytest

from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.diagnostics.features import COMPOSITE_ROBUST2, COMPOSITE_V1
from lsc.eval.detectors import make_composite_detector
from lsc.models import KalmanModel

NULL = AR1StateDGP(phi=0.95, q=0.04875, r=1.0)
BREAK = AR1StateDGP(phi=0.95, q=0.04875, r=1.0,
                    breaks=[BreakSpec(kind="variance", time_frac=0.5,
                                      vol_mult=1.5)])
T, N_TRAIN = 500, 125


def _build(include):
    return make_composite_detector(lambda: KalmanModel("ar1"), NULL, T,
                                   N_TRAIN, window=20, n_scale_reps=50,
                                   include=include)


@pytest.mark.parametrize("include", [COMPOSITE_V1, COMPOSITE_ROBUST2])
def test_composite_build_is_deterministic(include):
    Y = BREAK.sample(T, seed=200_007).Y
    s0, s1 = _build(include)(Y), _build(include)(Y)
    assert np.array_equal(np.nan_to_num(s0), np.nan_to_num(s1))


@pytest.mark.parametrize("include,t,golden", [
    (COMPOSITE_ROBUST2, 400, 8.891527),
    (COMPOSITE_ROBUST2, 499, 19.181717),
    (COMPOSITE_V1, 400, 20.225122),
    (COMPOSITE_V1, 499, 29.926148),
])
def test_composite_golden_score(include, t, golden):
    # pins the post-§8.4 per-time-point-standardized composite output;
    # a change here means artifacts must be regenerated (see CHANGELOG).
    s = _build(include)(BREAK.sample(T, seed=200_007).Y)
    assert s[t] == pytest.approx(golden, abs=1e-3)


def test_exceedance_feature_alive_inside_composite():
    # robust2 (exceedance features in place of e^2) must register the
    # subtle x1.5 break, not sit at chance — the "diluted, not dead"
    # correction to paper §8.3(ii).
    s = _build(COMPOSITE_ROBUST2)(BREAK.sample(T, seed=200_007).Y)
    assert s[499] > 5.0
