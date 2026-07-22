"""exp14_mixed_channel.py -- tests whether "run both a raw and a
whitened variance CUSUM" (Sec 10's original practical recommendation)
actually holds up when the break channel (r vs q) is unknown to the
detector, as it would be in practice.

Reviewer-requested experiment (peer review round 2, Missing Experiment
#1 / Question 4): "Is the practical recommendation validated anywhere
against a mixed-channel DGP, or is it an extrapolation from two
single-channel grids?" It was the latter -- this fills that gap.

Design: for each replicate, the channel (r or q) is chosen uniformly
at random and unknown to the detector -- exactly the "practitioner
doesn't know which variance moved" scenario the recommendation is
meant to address. Both raw_var_cusum and arima_var_cusum are run
regardless of true channel. Critically, the combined "run both"
detector is calibrated JOINTLY to hold a single 5% FAR budget (max of
each score's ratio to its own individually-calibrated threshold, that
combined ratio itself calibrated at 5% FAR) -- not run independently
at their own 5% each, which would silently compound the false-alarm
rate above 5%.

The q-channel break kind in this repo's BreakSpec is "state_var" (see
lsc/dgp/breaks.py) -- not "q_variance"; that was an earlier
independent-scaffold naming that silently errors or no-ops against the
real DGP if left in.

Usage: python experiments/exp14_mixed_channel.py [n_eval]
Output: prints calibration thresholds and detection rates per SNR to
stdout, and writes paper_assets/exp14_mixed_channel.csv.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "paper_assets" / "exp14_mixed_channel.csv"

PHI, R, T, N_TRAIN = 0.95, 1.0, 500, 125
VOL_MULT = 1.5


def run_snr(snr: float, n_eval: int, n_cal: int = 400, seed: int = 555) -> dict:
    q = snr * (1 - PHI**2) * R
    null_dgp = AR1StateDGP(phi=PHI, q=q, r=R)
    raw_fn = make_raw_var_cusum_detector(N_TRAIN)
    arima_fn = make_arima_var_cusum_detector(N_TRAIN)

    det_raw = calibrate("raw_var_cusum", raw_fn, null_dgp, T, n_reps=n_cal,
                         far=0.05, seed0=100_000)
    det_arima = calibrate("arima_var_cusum", arima_fn, null_dgp, T, n_reps=n_cal,
                           far=0.05, seed0=200_000)

    def combined_ratio(Y):
        r = raw_fn(Y) / det_raw.threshold
        a = arima_fn(Y) / det_arima.threshold
        return np.fmax(r, a)

    null_maxes = np.empty(n_cal)
    for i in range(n_cal):
        Y = null_dgp.sample(T, seed=300_000 + i).Y
        s = combined_ratio(Y)
        null_maxes[i] = s[np.isfinite(s)].max()
    combined_threshold = float(np.quantile(null_maxes, 0.95))
    combined_empirical_far = float((null_maxes >= combined_threshold).mean())

    rng = np.random.default_rng(seed)
    hits_raw = hits_arima = hits_combined = 0
    for i in range(n_eval):
        channel = rng.choice(["r", "q"])
        kind = "variance" if channel == "r" else "state_var"
        dgp = AR1StateDGP(phi=PHI, q=q, r=R,
                          breaks=[BreakSpec(kind=kind, time_frac=0.5, vol_mult=VOL_MULT)])
        Y = dgp.sample(T, seed=500_000 + i).Y
        ar = det_raw.alarm_time(Y)
        aa = det_arima.alarm_time(Y)
        s_comb = combined_ratio(Y)
        idx = np.where(s_comb[N_TRAIN:] >= combined_threshold)[0]
        ac = int(idx[0] + N_TRAIN) if len(idx) else None
        break_time = dgp.breaks[0].time(T)
        hits_raw += ar is not None and ar >= break_time
        hits_arima += aa is not None and aa >= break_time
        hits_combined += ac is not None and ac >= break_time

    return dict(snr=snr, n_cal=n_cal, n_eval=n_eval,
                threshold_raw=det_raw.threshold, threshold_arima=det_arima.threshold,
                threshold_combined=combined_threshold,
                empirical_far_raw=float((det_raw.null_max_scores >= det_raw.threshold).mean()),
                empirical_far_arima=float((det_arima.null_max_scores >= det_arima.threshold).mean()),
                empirical_far_combined=combined_empirical_far,
                raw=hits_raw / n_eval, arima=hits_arima / n_eval,
                combined=hits_combined / n_eval)


if __name__ == "__main__":
    n_eval = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    t0 = time.time()
    print(f"Mixed-channel test (50/50 r/q, unknown to detector), n_eval={n_eval}:")
    rows = []
    for snr in (0.1, 0.5, 2.0):
        out = run_snr(snr, n_eval)
        rows.append(out)
        print(f"  SNR={out['snr']}: threshold raw={out['threshold_raw']:.3f} "
              f"arima={out['threshold_arima']:.3f} combined={out['threshold_combined']:.3f} | "
              f"empirical FAR raw={out['empirical_far_raw']:.3f} "
              f"arima={out['empirical_far_arima']:.3f} combined={out['empirical_far_combined']:.3f} | "
              f"detect raw={out['raw']:.3f} arima={out['arima']:.3f} combined={out['combined']:.3f}",
              flush=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"[{time.time()-t0:.0f}s] wrote {OUT_PATH}")
