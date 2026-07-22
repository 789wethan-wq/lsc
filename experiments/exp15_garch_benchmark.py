"""exp15_garch_benchmark.py -- runs the raw/ARIMA/GARCH variance-CUSUM
comparison through the actual pipeline (lsc.dgp, lsc.diagnostics.alarms,
lsc.eval.detectors, plus experiments.garch_detector for the new rung),
at the r-channel (observation-noise) and q-channel (state-innovation,
BreakSpec kind="state_var") checkpoints used elsewhere in the paper:
SNR in {0.5, 2.0}, vol_mult=1.5, T=500, n_train=125.

Usage: python experiments/exp15_garch_benchmark.py [n_reps]
Output: prints calibration thresholds, empirical FARs, and detection
rates per cell to stdout; writes paper_assets/exp15_garch_benchmark.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.diagnostics.alarms import calibrate
from lsc.eval.detectors import make_arima_var_cusum_detector, make_raw_var_cusum_detector
from garch_detector import make_garch_var_cusum_detector

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "paper_assets" / "exp15_garch_benchmark.csv"

PHI, R, T, N_TRAIN = 0.95, 1.0, 500, 125
FAR = 0.05
SEED_CAL, SEED_EVAL = 100_000, 200_000
VOL_MULT = 1.5


def run_cell(snr: float, channel: str, n_reps: int) -> dict:
    q = snr * (1 - PHI**2) * R
    null_dgp = AR1StateDGP(phi=PHI, q=q, r=R)

    raw_fn = make_raw_var_cusum_detector(N_TRAIN)
    arima_fn = make_arima_var_cusum_detector(N_TRAIN)
    garch_fn = make_garch_var_cusum_detector(N_TRAIN)

    det_raw = calibrate("raw_var_cusum", raw_fn, null_dgp, T, n_reps=n_reps,
                         far=FAR, seed0=SEED_CAL)
    det_arima = calibrate("arima_var_cusum", arima_fn, null_dgp, T, n_reps=n_reps,
                           far=FAR, seed0=SEED_CAL)
    det_garch = calibrate("garch_var_cusum", garch_fn, null_dgp, T, n_reps=n_reps,
                           far=FAR, seed0=SEED_CAL)

    kind = "variance" if channel == "r" else "state_var"
    break_dgp = AR1StateDGP(phi=PHI, q=q, r=R,
                             breaks=[BreakSpec(kind=kind, time_frac=0.5, vol_mult=VOL_MULT)])
    break_time = break_dgp.breaks[0].time(T)
    paths = [break_dgp.sample(T, seed=SEED_EVAL + i).Y for i in range(n_reps)]

    def detect_rate(det):
        return np.mean([det.alarm_time(Y) is not None and det.alarm_time(Y) >= break_time
                        for Y in paths])

    return dict(
        channel=channel, snr=snr, vol_mult=VOL_MULT, n_reps=n_reps,
        threshold_raw=det_raw.threshold, threshold_arima=det_arima.threshold,
        threshold_garch=det_garch.threshold,
        empirical_far_raw=float((det_raw.null_max_scores >= det_raw.threshold).mean()),
        empirical_far_arima=float((det_arima.null_max_scores >= det_arima.threshold).mean()),
        empirical_far_garch=float((det_garch.null_max_scores >= det_garch.threshold).mean()),
        detect_raw=detect_rate(det_raw), detect_arima=detect_rate(det_arima),
        detect_garch=detect_rate(det_garch),
    )


if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    t0 = time.time()
    rows = []
    for channel in ("r", "q"):
        for snr in (0.5, 2.0):
            out = run_cell(snr, channel, n_reps)
            rows.append(out)
            print(f"[{time.time()-t0:6.0f}s] channel={channel} SNR={snr} "
                  f"vol_mult={VOL_MULT} n_reps={n_reps}:")
            print(f"  threshold   raw={out['threshold_raw']:.3f}  "
                  f"arima={out['threshold_arima']:.3f}  garch={out['threshold_garch']:.3f}")
            print(f"  emp. FAR    raw={out['empirical_far_raw']:.3f}  "
                  f"arima={out['empirical_far_arima']:.3f}  garch={out['empirical_far_garch']:.3f}")
            print(f"  detect rate raw={out['detect_raw']:.3f}  "
                  f"arima={out['detect_arima']:.3f}  garch={out['detect_garch']:.3f}",
                  flush=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote {OUT_PATH}")
