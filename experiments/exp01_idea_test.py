"""Experiment 01 — first test of the core LSC idea.

Question: at a COMMON calibrated false-alarm rate (5% per 500 obs on the
matched null), do diagnostics on a filtered latent path (innovation
CUSUM, six-feature composite) detect structural breaks earlier / more
often than the raw-Y CUSUM and plain-HMM benchmarks?

Primary arena: mean-reverting AR(1) latent state (phi=0.95, SNR 0.5).
Here a level shift in the state is persistent relative to the model's
dynamics, so it is identifiable online — and raw-Y CUSUM is a fair,
competitive benchmark because Y is stationary pre-break.

Secondary arena (hard case, documented): random-walk local level. The
smoke study showed the well-specified Kalman filter absorbs an abrupt
level jump as an ordinary random-walk shock within a few steps, so no
causal method has power against small breaks there; raw CUSUM is
meaningless on a nonstationary path. Reported for honesty, not as the
headline comparison.

Design (SPEC §4 fully respected):
  - T=500, train prefix 125, break at t=250.
  - All detectors calibrated on the SAME 500 null draws (seeds
    100_000+), FAR target 5% per T obs.
  - Empirical FAR re-checked on 500 FRESH null draws (seeds 300_000+).
  - Evaluation on 500 break draws per scenario (seeds 200_000+); every
    method sees identical draws.
Seed ranges disjoint by construction; composite feature scales use
seeds 900_000+.

Usage: python experiments/exp01_idea_test.py [n_reps]
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from lsc.dgp import AR1StateDGP, BreakSpec, LocalLevelDGP
from lsc.diagnostics.alarms import calibrate, empirical_far
from lsc.eval.detectors import (
    make_arima_cusum_detector,
    make_composite_detector,
    make_innovation_cusum_detector,
    make_plain_hmm_detector,
    make_raw_cusum_detector,
    make_state_cusum_detector,
)
from lsc.eval.metrics import detection_outcome, summarize_detection
from lsc.models import KalmanModel

T = 500
N_TRAIN = 125
FAR_TARGET = 0.05
BREAK_FRAC = 0.5

ARENAS = {
    "ar1": dict(
        null=AR1StateDGP(phi=0.95, q=0.5, r=1.0),
        make_break_dgp=lambda breaks: AR1StateDGP(phi=0.95, q=0.5, r=1.0,
                                                  breaks=breaks),
        kalman_spec="ar1",
    ),
    "local_level": dict(
        null=LocalLevelDGP(q=0.5, r=1.0),
        make_break_dgp=lambda breaks: LocalLevelDGP(q=0.5, r=1.0, breaks=breaks),
        kalman_spec="llevel",
    ),
}

SCENARIOS = {
    "abrupt_level_0.5s": [BreakSpec("level", BREAK_FRAC, magnitude=0.5)],
    "abrupt_level_1s": [BreakSpec("level", BREAK_FRAC, magnitude=1.0)],
    "abrupt_level_3s": [BreakSpec("level", BREAK_FRAC, magnitude=3.0)],
    "ramp_1s_hl25": [BreakSpec("ramp", BREAK_FRAC, magnitude=1.0, half_life=25)],
    "variance_x3": [BreakSpec("variance", BREAK_FRAC, vol_mult=3.0)],
}


def build_detectors(arena: dict, n_reps: int) -> dict:
    spec = arena["kalman_spec"]
    return {
        "lsc_kalman_cusum": make_innovation_cusum_detector(
            lambda: KalmanModel(spec), N_TRAIN),
        "lsc_state_cusum": make_state_cusum_detector(
            lambda: KalmanModel(spec), N_TRAIN),
        "lsc_composite": make_composite_detector(
            lambda: KalmanModel(spec), arena["null"], T, N_TRAIN,
            n_scale_reps=min(50, n_reps)),
        "raw_cusum": make_raw_cusum_detector(N_TRAIN),
        "arima_cusum": make_arima_cusum_detector(N_TRAIN),
        "plain_hmm": make_plain_hmm_detector(N_TRAIN),
    }


def main(n_reps: int = 500) -> None:
    t0 = time.time()
    far_rows, rows = [], []
    for arena_name, arena in ARENAS.items():
        detectors = build_detectors(arena, n_reps)
        print(f"[{time.time()-t0:6.0f}s] === arena {arena_name}: detectors built ===",
              flush=True)

        calibrated = {}
        for name, fn in detectors.items():
            det = calibrate(name, fn, arena["null"], T, n_reps=n_reps,
                            far=FAR_TARGET, seed0=100_000)
            far = empirical_far(det, arena["null"], T, n_reps=n_reps,
                                seed0=300_000)
            calibrated[name] = det
            far_rows.append(dict(arena=arena_name, method=name,
                                 threshold=det.threshold,
                                 far_target=FAR_TARGET, far_empirical=far))
            print(f"[{time.time()-t0:6.0f}s] {name}: thr={det.threshold:.3f} "
                  f"empirical FAR={far:.3%} (target {FAR_TARGET:.0%})", flush=True)

        for scen_name, breaks in SCENARIOS.items():
            dgp = arena["make_break_dgp"](breaks)
            break_time = breaks[0].time(T)
            for det_name, det in calibrated.items():
                outcomes = []
                for i in range(n_reps):
                    s = dgp.sample(T, seed=200_000 + i)
                    outcomes.append(detection_outcome(det.alarm_time(s.Y),
                                                      break_time, T))
                summ = summarize_detection(outcomes)
                rows.append(dict(arena=arena_name, scenario=scen_name,
                                 method=det_name, **summ))
                print(f"[{time.time()-t0:6.0f}s] {arena_name:12s} {scen_name:18s} "
                      f"{det_name:18s} detect={summ['detect_rate']:.2f} "
                      f"med_delay={summ['median_delay_detected']:.0f} "
                      f"mean_delay_cens={summ['mean_delay_censored']:.1f}",
                      flush=True)

    pd.DataFrame(far_rows).to_csv("paper_assets/exp01_far_calibration.csv",
                                  index=False)
    df = pd.DataFrame(rows)
    df.to_parquet("paper_assets/exp01_results.parquet", index=False)
    df.to_csv("paper_assets/exp01_results.csv", index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote paper_assets/exp01_results.*")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
