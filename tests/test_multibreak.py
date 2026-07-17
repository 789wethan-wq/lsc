"""Re-arm alarm protocol and event-level multi-break metrics (exp04)."""
import numpy as np

from lsc.benchmarks.changepoint import raw_cusum_score, windowed_raw_cusum_score
from lsc.diagnostics.alarms import CalibratedDetector
from lsc.diagnostics.features import break_pressure, windowed_break_pressure
from lsc.eval.metrics import multi_break_outcome, summarize_multi_break


def det_for(score: np.ndarray, threshold: float = 1.0) -> CalibratedDetector:
    return CalibratedDetector(name="fake", score_fn=lambda Y: score,
                              threshold=threshold,
                              null_max_scores=np.zeros(1))


def test_alarm_times_rearm_after_drain():
    # crossing at 5, drains below 0.5 at 10, refractory passed, second
    # crossing at 40
    score = np.zeros(60)
    score[5:9] = 2.0
    score[9:40] = 0.1
    score[40:] = 3.0
    d = det_for(score)
    assert d.alarm_times(score, refractory=20) == [5, 40]
    assert d.alarm_times(score)[0] == d.alarm_time(score)


def test_no_rearm_while_saturated():
    # score never drains below rearm level -> only the first alarm fires
    score = np.zeros(100)
    score[10:] = 2.0
    d = det_for(score)
    assert d.alarm_times(score) == [10]


def test_refractory_blocks_early_rearm():
    # drains immediately, but second crossing arrives before refractory
    score = np.zeros(50)
    score[5] = 2.0
    score[6:15] = 0.0
    score[15] = 2.0   # only 10 obs after the alarm
    score[16:30] = 0.0
    score[30] = 2.0   # 25 obs after the alarm -> armed again
    d = det_for(score)
    assert d.alarm_times(score, refractory=20) == [5, 30]


def test_nan_warmup_skipped():
    score = np.full(30, np.nan)
    score[20:] = 2.0
    d = det_for(score)
    assert d.alarm_times(score) == [20]
    assert det_for(np.full(10, np.nan)).alarm_times(np.zeros(10)) == []


def test_multi_break_perfect_match():
    o = multi_break_outcome([205, 355], [200, 350], T=500, window=100)
    assert (o["tp"], o["fp"], o["fn"]) == (2, 0, 0)
    assert o["precision"] == 1.0 and o["recall"] == 1.0 and o["f1"] == 1.0
    assert o["mean_matched_delay"] == 5.0


def test_multi_break_one_to_one_greedy():
    # two alarms in the first break's window: first matches, second is FP
    o = multi_break_outcome([201, 210], [200, 350], T=500, window=100)
    assert (o["tp"], o["fp"], o["fn"]) == (1, 1, 1)
    assert o["recall"] == 0.5
    assert o["f1"] == 2 * 1 / (2 * 1 + 1 + 1)


def test_multi_break_pre_break_and_late_alarms_are_fp():
    # alarm before the first break and alarm outside any window: both FP
    o = multi_break_outcome([100, 470], [200, 350], T=500, window=100)
    assert (o["tp"], o["fp"], o["fn"]) == (0, 2, 2)
    assert o["f1"] == 0.0
    assert np.isnan(o["mean_matched_delay"])


def test_multi_break_no_alarms():
    o = multi_break_outcome([], [200, 350], T=500)
    assert o["n_alarms"] == 0 and o["recall"] == 0.0 and o["f1"] == 0.0
    assert np.isnan(o["precision"])


def test_summarize_multi_break_nan_aware():
    outs = [multi_break_outcome([], [200], T=500),
            multi_break_outcome([210], [200], T=500)]
    s = summarize_multi_break(outs)
    assert s["recall"] == 0.5
    assert s["precision"] == 1.0  # NaN excluded
    assert s["n"] == 2


def test_windowed_break_pressure_drains_after_permanent_shift():
    # a permanent +2 innovation-mean shift at t=100 in an otherwise-null
    # series: the cumulative statistic must stay pinned near its peak
    # (never drains -> exp04's second-event miss), the windowed variant
    # must fall back near its pre-shift level once the shift exits the
    # trailing window.
    rng = np.random.default_rng(0)
    e = rng.normal(size=400)
    e[100:] += 2.0
    cum = break_pressure(e, k=0.5, warmup=10)
    win = windowed_break_pressure(e, window=60, warmup=10)
    assert cum[380:400].mean() > cum[180:200].mean()  # cumulative: never drains
    # windowed: near-peak contrast around t=190 (test window mostly
    # post-shift, ref window still pre-shift), back near its null scale
    # (~unit z) by t~=380-400 once both windows sit in the new regime
    assert win[380:400].mean() < 0.3 * win[180:200].mean()


def test_windowed_raw_cusum_drains_after_permanent_shift():
    rng = np.random.default_rng(1)
    Y = rng.normal(size=400)
    Y[100:] += 2.0
    n_train = 50
    cum = raw_cusum_score(Y, n_train=n_train, k=0.5)
    win = windowed_raw_cusum_score(Y, n_train=n_train, window=60)
    assert cum[380:400].mean() > cum[180:200].mean()
    assert win[380:400].mean() < 0.3 * win[180:200].mean()


def test_windowed_break_pressure_still_detects_a_single_break():
    # bounded memory should not come at the cost of missing an isolated
    # break within the window
    rng = np.random.default_rng(2)
    e = rng.normal(size=200)
    e[100:] += 3.0
    win = windowed_break_pressure(e, window=60, warmup=10)
    assert np.nanmax(win[100:130]) > 5.0
