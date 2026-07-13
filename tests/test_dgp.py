"""M1 integrity tests: ground truth correctness and seed reproducibility
(SPEC §9 test_dgp_ground_truth)."""
import numpy as np
import pytest

from lsc.dgp import (
    AR1StateDGP,
    BreakSpec,
    LocalLevelDGP,
    LocalLinearTrendDGP,
    MarkovSwitchingDGP,
    TimeVaryingVolDGP,
    matched_null,
)

ALL_DGPS = [
    LocalLevelDGP(q=0.5, r=1.0),
    LocalLevelDGP(q=0.5, r=1.0, t_dof=5),
    LocalLevelDGP(q=0.5, r=1.0, breaks=[BreakSpec("level", 0.5, magnitude=1.0)]),
    LocalLevelDGP(q=0.5, r=1.0, breaks=[BreakSpec("variance", 0.5, vol_mult=3.0)]),
    LocalLevelDGP(q=0.5, r=1.0, breaks=[BreakSpec("ramp", 0.5, magnitude=1.0, half_life=25)]),
    LocalLevelDGP(q=0.5, r=1.0, breaks=[BreakSpec("level", f, magnitude=1.0) for f in (0.25, 0.5, 0.75)]),
    LocalLinearTrendDGP(),
    AR1StateDGP(phi=0.95, q=0.5, r=1.0),
    AR1StateDGP(phi=0.95, q=0.5, r=1.0, drift_coef=0.1),
    TimeVaryingVolDGP(breaks=[BreakSpec("variance", 0.5, vol_mult=3.0)]),
    MarkovSwitchingDGP(means=(0.0, 2.0), sigmas=(1.0, 1.0), persistence=0.98),
    MarkovSwitchingDGP(means=(0.0, 2.0, -2.0), sigmas=(1.0, 2.0, 1.0), persistence=0.95),
]


@pytest.mark.parametrize("dgp", ALL_DGPS, ids=lambda d: d.name + str(id(d) % 997))
def test_seed_reproducibility(dgp):
    a = dgp.sample(300, seed=42)
    b = dgp.sample(300, seed=42)
    np.testing.assert_array_equal(a.Y, b.Y)
    np.testing.assert_array_equal(a.S_true, b.S_true)
    assert a.break_times == b.break_times
    c = dgp.sample(300, seed=43)
    assert not np.array_equal(a.Y, c.Y)


@pytest.mark.parametrize("dgp", ALL_DGPS, ids=lambda d: d.name + str(id(d) % 997))
def test_shapes_and_finiteness(dgp):
    s = dgp.sample(200, seed=1)
    assert s.Y.shape == (200,)
    assert s.S_true.shape == (200,)
    assert np.all(np.isfinite(s.Y))
    assert np.all(np.isfinite(s.S_true))


def test_break_times_match_config():
    T = 500
    dgp = LocalLevelDGP(breaks=[BreakSpec("level", 0.5, magnitude=2.0)])
    s = dgp.sample(T, seed=7)
    assert s.break_times == [250]

    multi = LocalLevelDGP(breaks=[BreakSpec("level", f) for f in (0.2, 0.5, 0.8)])
    assert multi.sample(T, seed=7).break_times == [100, 250, 400]


def test_level_break_shifts_state():
    """The post-break state must be shifted by exactly magnitude*sigma_ref
    relative to the identically-seeded null path."""
    T = 500
    spec = BreakSpec("level", 0.5, magnitude=2.0)
    dgp = LocalLevelDGP(q=0.5, r=1.0, breaks=[spec])
    null = matched_null(dgp)
    s, s0 = dgp.sample(T, seed=3), null.sample(T, seed=3)
    delta = s.S_true - s0.S_true
    np.testing.assert_allclose(delta[:250], 0.0, atol=1e-12)
    np.testing.assert_allclose(delta[250:], 2.0 * dgp.sigma_ref, atol=1e-12)


def test_variance_break_scales_noise():
    T = 400
    spec = BreakSpec("variance", 0.5, vol_mult=3.0)
    dgp = LocalLevelDGP(q=0.0, r=1.0, breaks=[spec])
    s = dgp.sample(T, seed=11)
    resid = s.Y - s.S_true
    pre, post = resid[:200].std(), resid[200:].std()
    assert 2.0 < post / pre < 4.0  # noisy estimate of 3.0


def test_state_var_break_scales_state_innovation_sd():
    """q-break scales the state-innovation SD by vol_mult (the same SD
    convention the r-break uses on obs noise), and shifts Y's
    autocorrelation — the structural q/r distinction (M2)."""
    T = 6000
    phi = 0.9
    spec = BreakSpec("state_var", 0.5, vol_mult=1.5)
    dgp = AR1StateDGP(phi=phi, q=0.2, r=0.0, breaks=[spec])
    s = dgp.sample(T, seed=13)
    w = s.S_true[1:] - phi * s.S_true[:-1]        # state innovation (r=0)
    pre, post = w[100:2900].std(), w[3100:].std()
    assert 1.35 < post / pre < 1.65               # SD x1.5, noisy estimate
    # r-break with the SAME vol_mult scales obs-noise SD by the same factor
    rspec = BreakSpec("variance", 0.5, vol_mult=1.5)
    rdgp = LocalLevelDGP(q=0.0, r=0.2, breaks=[rspec])
    rs = rdgp.sample(T, seed=13)
    resid = rs.Y - rs.S_true
    assert 1.35 < resid[3100:].std() / resid[100:2900].std() < 1.65


def test_state_var_null_matched_and_reproducible():
    dgp = AR1StateDGP(phi=0.95, q=0.05, r=1.0,
                      breaks=[BreakSpec("state_var", 0.5, vol_mult=3.0)])
    assert dgp.null_version().breaks == []
    a = dgp.sample(500, seed=7).Y
    b = dgp.sample(500, seed=7).Y
    assert np.array_equal(a, b)
    # a no-break DGP is bit-identical whether or not it carries the code path
    null = AR1StateDGP(phi=0.95, q=0.05, r=1.0)
    assert np.array_equal(null.sample(500, seed=7).Y,
                          dgp.null_version().sample(500, seed=7).Y)


def test_ramp_is_gradual_and_reaches_target():
    T = 1000
    spec = BreakSpec("ramp", 0.5, magnitude=1.0, half_life=25)
    dgp = LocalLevelDGP(q=0.5, r=1.0, breaks=[spec])
    null = matched_null(dgp)
    delta = dgp.sample(T, seed=5).S_true - null.sample(T, seed=5).S_true
    target = dgp.sigma_ref
    assert abs(delta[500] - 0.5 * target) < 1e-9      # half at center
    assert delta[999] > 0.99 * target                  # full at the end
    assert np.all(np.diff(delta) >= 0)                 # monotone


def test_markov_regime_ground_truth():
    dgp = MarkovSwitchingDGP(means=(0.0, 2.0), sigmas=(1.0, 1.0), persistence=0.9)
    s = dgp.sample(2000, seed=9)
    # break_times must be exactly where regime_path changes
    changes = list(np.nonzero(np.diff(s.regime_path))[0] + 1)
    assert s.break_times == changes
    assert len(changes) > 0
    # S_true equals the active regime's mean
    np.testing.assert_array_equal(s.S_true, np.asarray(dgp.means)[s.regime_path])


def test_null_versions_have_no_breaks():
    for dgp in ALL_DGPS:
        null = matched_null(dgp)
        s = null.sample(400, seed=2)
        assert s.break_times == []


def test_markov_null_never_switches():
    dgp = MarkovSwitchingDGP(persistence=0.9)
    s = matched_null(dgp).sample(1000, seed=4)
    assert np.all(s.regime_path == s.regime_path[0])


def test_heavy_tail_flag_changes_noise_only_scale_preserved():
    """t5 noise has the same variance as Gaussian (misspecification flag)."""
    dgp = LocalLevelDGP(q=0.0, r=1.0, t_dof=5)
    resid = np.concatenate([
        dgp.sample(2000, seed=s).Y for s in range(5)
    ])
    assert abs(resid.std() - 1.0) < 0.1


def test_persistence_break_preserves_marginals():
    """Persistence break is a pure dynamics change: stationary mean and
    variance of the state (and Y) are unchanged; autocorrelation rises."""
    from lsc.dgp import AR1StateDGP, BreakSpec

    T = 20000
    dgp = AR1StateDGP(phi=0.9, q=0.19, r=1.0,
                      breaks=[BreakSpec("persistence", 0.5, new_phi=0.99)])
    s = dgp.sample(T, seed=21)
    assert s.break_times == [T // 2]
    pre, post = s.S_true[2000:T // 2], s.S_true[T // 2 + 2000:]
    # same stationary sd (loose tolerance: persistent series, few
    # effective obs); autocorrelation must clearly increase
    assert abs(post.std() / pre.std() - 1.0) < 0.35
    ac = lambda x: np.corrcoef(x[:-1], x[1:])[0, 1]
    assert ac(pre) < 0.95 and ac(post) > 0.97


def test_persistence_break_reproducible_and_null_matched():
    from lsc.dgp import AR1StateDGP, BreakSpec, matched_null

    dgp = AR1StateDGP(phi=0.95, q=0.5, r=1.0,
                      breaks=[BreakSpec("persistence", 0.5, new_phi=0.995)])
    a, b = dgp.sample(500, seed=3), dgp.sample(500, seed=3)
    np.testing.assert_array_equal(a.Y, b.Y)
    # identical to the null path before the break (same seed)
    n = matched_null(dgp).sample(500, seed=3)
    np.testing.assert_array_equal(a.Y[:250], n.Y[:250])
    assert not np.array_equal(a.Y[250:], n.Y[250:])
