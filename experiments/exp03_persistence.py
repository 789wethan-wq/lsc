"""Experiment 03 — pure dynamics breaks (persistence changes).

The break leaves the stationary mean and variance of Y unchanged and
only shifts the AR(1) coefficient of the latent state (state-noise
variance rescaled accordingly). Hypothesis registered in CHANGELOG
before running: Y-space level detectors blind, ARIMA residual CUSUM
partially sighted, LSC diagnostics layer clearly above FAR.

Arena: AR(1), phi=0.95, spec-SNR 0.5 (q = 0.5*(1-phi^2)), T=500,
train 125, break at 250. Same harness/seed layout/FAR target as
exp01/exp02.

Usage: python experiments/exp03_persistence.py [n_reps]
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.diagnostics.alarms import calibrate, empirical_far
from lsc.eval.detectors import (
    make_arima_cusum_detector,
    make_composite_detector,
    make_innovation_cusum_detector,
    make_raw_cusum_detector,
    make_state_cusum_detector,
)
from lsc.eval.metrics import detection_outcome, summarize_detection
from lsc.models import KalmanModel

T = 500
N_TRAIN = 125
FAR_TARGET = 0.05
PHI = 0.95
R = 1.0
Q = 0.5 * R * (1.0 - PHI**2)  # spec-SNR 0.5

SCENARIOS = {
    "persistence_up_0.995": [BreakSpec("persistence", 0.5, new_phi=0.995)],
    "persistence_down_0.80": [BreakSpec("persistence", 0.5, new_phi=0.80)],
}


def main(n_reps: int = 500) -> None:
    t0 = time.time()
    null = AR1StateDGP(phi=PHI, q=Q, r=R)
    detectors = {
        "lsc_composite": make_composite_detector(
            lambda: KalmanModel("ar1"), null, T, N_TRAIN,
            n_scale_reps=min(50, n_reps)),
        "lsc_kalman_cusum": make_innovation_cusum_detector(
            lambda: KalmanModel("ar1"), N_TRAIN),
        "lsc_state_cusum": make_state_cusum_detector(
            lambda: KalmanModel("ar1"), N_TRAIN),
        "raw_cusum": make_raw_cusum_detector(N_TRAIN),
        "arima_cusum": make_arima_cusum_detector(N_TRAIN),
    }
    far_rows, rows = [], []
    calibrated = {}
    for name, fn in detectors.items():
        det = calibrate(name, fn, null, T, n_reps=n_reps, far=FAR_TARGET,
                        seed0=100_000)
        far = empirical_far(det, null, T, n_reps=n_reps, seed0=300_000)
        calibrated[name] = det
        far_rows.append(dict(method=name, threshold=det.threshold,
                             far_target=FAR_TARGET, far_empirical=far))
        print(f"[{time.time()-t0:6.0f}s] {name}: thr={det.threshold:.3f} "
              f"FAR={far:.3%}", flush=True)

    for scen_name, breaks in SCENARIOS.items():
        dgp = AR1StateDGP(phi=PHI, q=Q, r=R, breaks=breaks)
        break_time = breaks[0].time(T)
        for det_name, det in calibrated.items():
            outcomes = [
                detection_outcome(
                    det.alarm_time(dgp.sample(T, seed=200_000 + i).Y),
                    break_time, T)
                for i in range(n_reps)
            ]
            summ = summarize_detection(outcomes)
            rows.append(dict(scenario=scen_name, method=det_name, **summ))
            print(f"[{time.time()-t0:6.0f}s] {scen_name:22s} {det_name:18s} "
                  f"detect={summ['detect_rate']:.2f} "
                  f"med_delay={summ['median_delay_detected']:.0f}", flush=True)

    pd.DataFrame(far_rows).to_csv("paper_assets/exp03_far_calibration.csv",
                                  index=False)
    df = pd.DataFrame(rows)
    df.to_parquet("paper_assets/exp03_results.parquet", index=False)
    df.to_csv("paper_assets/exp03_results.csv", index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote paper_assets/exp03_results.*")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
