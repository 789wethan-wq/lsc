"""Experiment 03b — does the quietness-upgraded composite detect pure
dynamics breaks? (Hypothesis in CHANGELOG, registered before running.)

Identical arena/seeds/scenarios to exp03; only the composite detector
(now 11 features) is recalibrated and evaluated. Benchmarks are
unchanged code, so exp03's numbers remain valid for them. Also
evaluates the composite on exp01's level/variance scenarios to measure
the breadth tax of the added features.

Usage: python experiments/exp03b_quietness.py [n_reps]
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.diagnostics.alarms import calibrate, empirical_far
from lsc.eval.detectors import make_composite_detector
from lsc.eval.metrics import detection_outcome, summarize_detection
from lsc.models import KalmanModel

T = 500
N_TRAIN = 125
FAR_TARGET = 0.05
PHI = 0.95
R = 1.0
Q = 0.5 * R * (1.0 - PHI**2)  # spec-SNR 0.5 (same as exp03)

SCENARIOS = {
    "persistence_up_0.995": [BreakSpec("persistence", 0.5, new_phi=0.995)],
    "persistence_down_0.80": [BreakSpec("persistence", 0.5, new_phi=0.80)],
    # breadth-tax re-measurement on exp02-style scenarios (same arena)
    "abrupt_level_3s": [BreakSpec("level", 0.5, magnitude=3.0)],
    "variance_x3": [BreakSpec("variance", 0.5, vol_mult=3.0)],
}


def main(n_reps: int = 500) -> None:
    t0 = time.time()
    null = AR1StateDGP(phi=PHI, q=Q, r=R)
    fn = make_composite_detector(lambda: KalmanModel("ar1"), null, T, N_TRAIN,
                                 n_scale_reps=min(50, n_reps))
    det = calibrate("lsc_composite_v2", fn, null, T, n_reps=n_reps,
                    far=FAR_TARGET, seed0=100_000)
    far = empirical_far(det, null, T, n_reps=n_reps, seed0=300_000)
    print(f"[{time.time()-t0:6.0f}s] lsc_composite_v2: thr={det.threshold:.3f} "
          f"FAR={far:.3%}", flush=True)
    pd.DataFrame([dict(method="lsc_composite_v2", threshold=det.threshold,
                       far_target=FAR_TARGET, far_empirical=far)]
                 ).to_csv("paper_assets/exp03b_far_calibration.csv", index=False)

    rows = []
    for scen_name, breaks in SCENARIOS.items():
        dgp = AR1StateDGP(phi=PHI, q=Q, r=R, breaks=breaks)
        break_time = breaks[0].time(T)
        outcomes = [
            detection_outcome(det.alarm_time(dgp.sample(T, seed=200_000 + i).Y),
                              break_time, T)
            for i in range(n_reps)
        ]
        summ = summarize_detection(outcomes)
        rows.append(dict(scenario=scen_name, method="lsc_composite_v2", **summ))
        print(f"[{time.time()-t0:6.0f}s] {scen_name:22s} "
              f"detect={summ['detect_rate']:.2f} "
              f"med_delay={summ['median_delay_detected']:.0f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet("paper_assets/exp03b_results.parquet", index=False)
    df.to_csv("paper_assets/exp03b_results.csv", index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote paper_assets/exp03b_results.*")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
