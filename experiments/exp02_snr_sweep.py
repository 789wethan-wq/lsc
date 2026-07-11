"""Experiment 02 — where does latent-space detection beat raw-Y detection?

exp01's AR(1) arena had stationary-state-variance / obs-variance = 5.1
(the spec's SNR definition), i.e. observations were nearly noiseless —
the regime where filtering buys least, and raw-Y CUSUM matched the
latent state CUSUM. Mechanism-based hypothesis (stated BEFORE running):
as SNR falls, the Kalman filter removes a growing share of observation
noise, so the latent baseline CUSUM should gain a detection edge over
raw-Y CUSUM at matched FAR. This sweeps SNR in {0.1, 0.5, 2.0}.

Breaks are expressed in units of the stationary STATE sd (sigma_ref),
so a "1σ break" is comparably hidden at every SNR: at SNR 0.1 a 1σ
state shift is only 0.32 obs-noise-sd of displacement in Y.

Same harness, seed layout, FAR target, and rep counts as exp01.
Methods: the two head-to-head CUSUMs, the composite, and raw-Y CUSUM.
(ARIMA and plain-HMM dropped for runtime; exp01 showed both dominated.)

Usage: python experiments/exp02_snr_sweep.py [n_reps]
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.diagnostics.alarms import calibrate, empirical_far
from lsc.eval.detectors import (
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
SNRS = [0.1, 0.5, 2.0]  # stationary state var / obs var (SPEC §6 definition)

SCENARIOS = {
    "abrupt_level_1s": [BreakSpec("level", 0.5, magnitude=1.0)],
    "abrupt_level_3s": [BreakSpec("level", 0.5, magnitude=3.0)],
    "ramp_3s_hl25": [BreakSpec("ramp", 0.5, magnitude=3.0, half_life=25)],
}


def q_for_snr(snr: float) -> float:
    return snr * R * (1.0 - PHI**2)


def main(n_reps: int = 500) -> None:
    t0 = time.time()
    far_rows, rows = [], []
    for snr in SNRS:
        null = AR1StateDGP(phi=PHI, q=q_for_snr(snr), r=R)
        detectors = {
            "lsc_state_cusum": make_state_cusum_detector(
                lambda: KalmanModel("ar1"), N_TRAIN),
            "lsc_kalman_cusum": make_innovation_cusum_detector(
                lambda: KalmanModel("ar1"), N_TRAIN),
            "lsc_composite": make_composite_detector(
                lambda: KalmanModel("ar1"), null, T, N_TRAIN,
                n_scale_reps=min(50, n_reps)),
            "raw_cusum": make_raw_cusum_detector(N_TRAIN),
        }
        calibrated = {}
        for name, fn in detectors.items():
            det = calibrate(name, fn, null, T, n_reps=n_reps,
                            far=FAR_TARGET, seed0=100_000)
            far = empirical_far(det, null, T, n_reps=n_reps, seed0=300_000)
            calibrated[name] = det
            far_rows.append(dict(snr=snr, method=name, threshold=det.threshold,
                                 far_target=FAR_TARGET, far_empirical=far))
            print(f"[{time.time()-t0:6.0f}s] snr={snr} {name}: "
                  f"thr={det.threshold:.3f} FAR={far:.3%}", flush=True)

        for scen_name, breaks in SCENARIOS.items():
            dgp = AR1StateDGP(phi=PHI, q=q_for_snr(snr), r=R, breaks=breaks)
            break_time = breaks[0].time(T)
            for det_name, det in calibrated.items():
                outcomes = [
                    detection_outcome(
                        det.alarm_time(dgp.sample(T, seed=200_000 + i).Y),
                        break_time, T)
                    for i in range(n_reps)
                ]
                summ = summarize_detection(outcomes)
                rows.append(dict(snr=snr, scenario=scen_name,
                                 method=det_name, **summ))
                print(f"[{time.time()-t0:6.0f}s] snr={snr} {scen_name:16s} "
                      f"{det_name:18s} detect={summ['detect_rate']:.2f} "
                      f"med_delay={summ['median_delay_detected']:.0f}",
                      flush=True)

    pd.DataFrame(far_rows).to_csv("paper_assets/exp02_far_calibration.csv",
                                  index=False)
    df = pd.DataFrame(rows)
    df.to_parquet("paper_assets/exp02_results.parquet", index=False)
    df.to_csv("paper_assets/exp02_results.csv", index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote paper_assets/exp02_results.*")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
