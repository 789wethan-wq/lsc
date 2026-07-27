"""exp38 -- does raw_cusum's rising empirical FAR with SNR (Table 1:
4.0% / 6.2% / 8.2% at SNR 0.1/0.5/2.0) inflate its Table 2 detection-
rate advantage over the innovation CUSUM? (SPEC R6 M1, pre-registered
in experiments/CHANGELOG.md 2026-07-26 BEFORE this script was run,
design corrected after checking a premise that did not match the code)

PREMISE CHECK (done before writing this script): the request assumed
raw_cusum uses "a single pooled threshold" shared across SNRs. It does
not -- `lsc.eval.runner.run` already calls `calibrate()` once per
(arena, method), so grid_v1's raw_cusum threshold is ALREADY set
separately at each SNR from that SNR's own 500 fresh null draws
(verified: paper_assets/grid_v1_far_calibration.csv has three distinct
raw_cusum thresholds -- 27.49 / 103.19 / 213.89 at SNR 0.1/0.5/2.0, not
one shared value). So there is no pooled-threshold confound to remove.

The REAL, well-posed version of the reviewer's concern, given the
architecture is already per-SNR: is the empirical-FAR drift itself a
finite-sample threshold-ESTIMATION artifact -- does a much larger
calibration sample at each SNR converge toward a tighter threshold (and
closer-to-5% out-of-sample FAR), and if so, how much does raw_cusum's
detection rate change when scored against that more precisely
calibrated threshold instead of the original n_reps=500 one?

Design: for each SNR (grid_v1's arena params: phi=0.95, q=SNR*(1-phi^2),
r=1.0), recalibrate raw_cusum with n_reps=5000 (calibration seed0=100000
-- a superset of the original 500 draws, so the original threshold is
exactly what these same draws produce when truncated to the first 500,
not a different draw sequence). Compare: threshold_500 (reproduces the
published value) vs. threshold_5000; each threshold's out-of-sample FAR
on 2000 FRESH null draws (far_check=300000, disjoint from both
calibration blocks); and each threshold's detect_rate on the SAME
n_reps=500 evaluation draws (seed0=200000) Table 2 itself uses, for a
directly comparable number. lsc_kalman_cusum (the innovation CUSUM
Table 2 compares against) is included at n_reps=500 only, unchanged,
since Table 1 shows it calibrating close to 5% at every SNR already
(no drift to investigate there).

Usage: python experiments/exp38_raw_cusum_far_correction.py [n_reps_large]
Output: paper_assets/exp38_raw_cusum_far_correction.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.diagnostics.alarms import calibrate, empirical_far
from lsc.eval.detectors import make_innovation_cusum_detector, make_raw_cusum_detector
from lsc.models import KalmanModel

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "paper_assets" / "exp38_raw_cusum_far_correction.csv"

PHI, R, T, N_TRAIN = 0.95, 1.0, 500, 125
FAR = 0.05
SEED_CAL, SEED_EVAL, SEED_FAR_CHECK = 100_000, 200_000, 300_000
N_REPS_EVAL = 500
N_REPS_FAR_CHECK = 2000

SNRS = (0.1, 0.5, 2.0)


def run_snr(snr: float, n_reps_large: int) -> dict:
    q = snr * (1 - PHI**2) * R
    null_dgp = AR1StateDGP(phi=PHI, q=q, r=R)
    raw_fn = make_raw_cusum_detector(N_TRAIN)

    det_500 = calibrate("raw_cusum", raw_fn, null_dgp, T, n_reps=N_REPS_EVAL,
                        far=FAR, seed0=SEED_CAL)
    det_large = calibrate("raw_cusum", raw_fn, null_dgp, T, n_reps=n_reps_large,
                          far=FAR, seed0=SEED_CAL)

    far_500_out = empirical_far(det_500, null_dgp, T, n_reps=N_REPS_FAR_CHECK,
                                seed0=SEED_FAR_CHECK)
    far_large_out = empirical_far(det_large, null_dgp, T, n_reps=N_REPS_FAR_CHECK,
                                  seed0=SEED_FAR_CHECK)

    kind = "level"
    break_dgp = AR1StateDGP(phi=PHI, q=q, r=R,
                            breaks=[BreakSpec(kind=kind, time_frac=0.5, magnitude=3.0)])
    break_time = break_dgp.breaks[0].time(T)
    paths = [break_dgp.sample(T, seed=SEED_EVAL + i).Y for i in range(N_REPS_EVAL)]

    def detect_rate(det):
        return float(np.mean([det.alarm_time(Y) is not None and det.alarm_time(Y) >= break_time
                              for Y in paths]))

    detect_500 = detect_rate(det_500)
    detect_large = detect_rate(det_large)

    kalman_fn = make_innovation_cusum_detector(lambda: KalmanModel("ar1"), N_TRAIN)
    det_kalman = calibrate("lsc_kalman_cusum", kalman_fn, null_dgp, T, n_reps=N_REPS_EVAL,
                           far=FAR, seed0=SEED_CAL)
    detect_kalman = detect_rate(det_kalman)

    return dict(
        snr=snr, n_reps_large=n_reps_large,
        threshold_500=det_500.threshold, threshold_large=det_large.threshold,
        far_500_calibration_draw=float((det_500.null_max_scores >= det_500.threshold).mean()),
        far_large_calibration_draw=float((det_large.null_max_scores >= det_large.threshold).mean()),
        far_500_out_of_sample=far_500_out, far_large_out_of_sample=far_large_out,
        detect_raw_500=detect_500, detect_raw_large=detect_large,
        detect_kalman=detect_kalman,
        raw_advantage_500=detect_500 - detect_kalman,
        raw_advantage_large=detect_large - detect_kalman,
    )


if __name__ == "__main__":
    n_reps_large = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    t0 = time.time()
    rows = []
    for snr in SNRS:
        out = run_snr(snr, n_reps_large)
        rows.append(out)
        print(f"[{time.time()-t0:6.0f}s] SNR={snr}: "
              f"thr500={out['threshold_500']:.2f} (out-of-sample FAR "
              f"{out['far_500_out_of_sample']:.3f}) "
              f"thr_large={out['threshold_large']:.2f} (out-of-sample FAR "
              f"{out['far_large_out_of_sample']:.3f}) | "
              f"detect raw@500={out['detect_raw_500']:.3f} "
              f"raw@large={out['detect_raw_large']:.3f} "
              f"kalman={out['detect_kalman']:.3f} | "
              f"advantage@500={out['raw_advantage_500']:+.3f} "
              f"advantage@large={out['raw_advantage_large']:+.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote {OUT_PATH}")
