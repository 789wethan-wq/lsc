"""exp22_composite_threshold_argmax.py -- MW3 diagnostic (peer review
round 3): at the r-channel x1.5/SNR-0.1 cell -- where exp20's
composite-on-ARIMA is worst relative to composite-on-Kalman (0.226 vs.
0.818, Table 8) -- report the CALIBRATED THRESHOLD and the
ARGMAX-FEATURE DISTRIBUTION under a break for both composites, to
distinguish two readings of exp20/exp21's Kalman-vs-ARIMA gap:

  (a) "destructive substitution" -- the 6 filtered-state-analog
      features carry real signal on Kalman inputs that is lost (not
      replaced by noise) when fed ARIMA inputs;
  (b) "noisy substitute" -- the 6 ARIMA-fed state-analog features
      actively inflate the composite's null score distribution, which
      raises the calibrated threshold and swamps whatever signal
      the 5 shared innovation-only features still carry.

If the ARIMA composite's calibrated threshold is substantially higher
than the Kalman composite's, that supports (b). Reconstructs both
composites with the EXACT recipe already used to produce the published
numbers (grid_v1's lsc_composite for Kalman: seeds.calibration=100000,
n_scale_reps=50, scale_seed0=900000, T=500, n_train=125; exp20's
composite-on-ARIMA: identical seeds, same construction, ARIMAModel in
place of KalmanModel) so the thresholds reported here are the same
numbers underlying Table 8, not a fresh recalibration.

Usage: python experiments/exp22_composite_threshold_argmax.py [n_eval]
Output: prints thresholds + argmax-feature histograms; writes
paper_assets/exp22_composite_threshold_argmax.csv (per-path attribution)
and paper_assets/exp22_summary.csv (aggregated).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.diagnostics.alarms import calibrate
from lsc.diagnostics.features import COMPOSITE_V1, compute_features
from lsc.eval.detectors import make_composite_detector
from lsc.models import ARIMAModel, KalmanModel

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "paper_assets" / "exp22_composite_threshold_argmax.csv"
SUMMARY_PATH = REPO_ROOT / "paper_assets" / "exp22_summary.csv"

PHI, R, T, N_TRAIN = 0.95, 1.0, 500, 125
SNR, VOL_MULT = 0.1, 1.5
SEED_CAL, SEED_EVAL, SEED_SCALE = 100_000, 200_000, 900_000


def composite_attribution(score_fn, Y: np.ndarray, t: int) -> str:
    """Name of the feature whose standardized z is maximal at index t
    (same definition as experiments/real_data.py's alarm attribution)."""
    est = score_fn.model_factory().fit_filter(Y, n_train=score_fn.n_train)
    feats = compute_features(est, window=score_fn.window, n_train=score_fn.n_train)
    best, best_z = "?", -np.inf
    for name in score_fn.include:
        c, s = score_fn.scales[name]
        z = abs(feats[name][t] - c[t]) / s[t]
        if np.isfinite(z) and z > best_z:
            best, best_z = name, z
    return best


def run(n_eval: int) -> None:
    t0 = time.time()
    q = SNR * (1 - PHI**2) * R
    null_dgp = AR1StateDGP(phi=PHI, q=q, r=R)

    fn_kalman = make_composite_detector(lambda: KalmanModel("ar1"), null_dgp, T, N_TRAIN,
                                        n_scale_reps=50, scale_seed0=SEED_SCALE,
                                        include=COMPOSITE_V1)
    det_kalman = calibrate("composite_kalman", fn_kalman, null_dgp, T,
                           n_reps=500, far=0.05, seed0=SEED_CAL)

    fn_arima = make_composite_detector(lambda: ARIMAModel(), null_dgp, T, N_TRAIN,
                                       n_scale_reps=50, scale_seed0=SEED_SCALE,
                                       include=COMPOSITE_V1)
    det_arima = calibrate("composite_arima", fn_arima, null_dgp, T,
                          n_reps=500, far=0.05, seed0=SEED_CAL)

    print(f"[{time.time()-t0:.0f}s] threshold_kalman={det_kalman.threshold:.3f} "
          f"threshold_arima={det_arima.threshold:.3f} "
          f"(ratio arima/kalman = {det_arima.threshold/det_kalman.threshold:.3f})",
          flush=True)

    break_dgp = AR1StateDGP(phi=PHI, q=q, r=R,
                            breaks=[BreakSpec(kind="variance", time_frac=0.5, vol_mult=VOL_MULT)])
    break_time = break_dgp.breaks[0].time(T)

    rows = []
    for i in range(n_eval):
        Y = break_dgp.sample(T, seed=SEED_EVAL + i).Y
        for tag, det, fn in (("kalman", det_kalman, fn_kalman), ("arima", det_arima, fn_arima)):
            alarm_t = det.alarm_time(Y)
            hit = alarm_t is not None and alarm_t >= break_time
            attrib_t = alarm_t if alarm_t is not None else int(
                np.nanargmax(np.nan_to_num(fn(Y)[N_TRAIN:], nan=-np.inf)) + N_TRAIN)
            feat = composite_attribution(fn, Y, attrib_t)
            rows.append(dict(model=tag, rep=i, alarmed=alarm_t is not None, hit=hit,
                             alarm_time=alarm_t, attrib_time=attrib_t, attrib_feature=feat))
        if (i + 1) % 50 == 0:
            print(f"[{time.time()-t0:.0f}s] {i+1}/{n_eval} paths done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    summary_rows = []
    for tag in ("kalman", "arima"):
        sub = df[df.model == tag]
        alarmed = sub[sub.alarmed]
        print(f"\n=== {tag} (n_eval={n_eval}, alarmed={len(alarmed)}/{len(sub)}, "
              f"hit_rate={sub.hit.mean():.3f}) ===")
        print("argmax-at-alarm-time feature distribution (alarmed paths only):")
        counts = alarmed.attrib_feature.value_counts()
        print(counts.to_string())
        for feat, cnt in counts.items():
            summary_rows.append(dict(model=tag, feature=feat, n_alarmed_attrib=int(cnt),
                                     frac_of_alarmed=float(cnt / max(len(alarmed), 1))))
        print("argmax-at-max-score-time feature distribution (ALL paths, alarmed or not):")
        counts_all = sub.attrib_feature.value_counts()
        print(counts_all.to_string())

    threshold_row = dict(threshold_kalman=det_kalman.threshold, threshold_arima=det_arima.threshold,
                         threshold_ratio_arima_over_kalman=det_arima.threshold / det_kalman.threshold,
                         empirical_far_kalman=float((det_kalman.null_max_scores >= det_kalman.threshold).mean()),
                         empirical_far_arima=float((det_arima.null_max_scores >= det_arima.threshold).mean()))
    pd.DataFrame([threshold_row]).to_csv(
        SUMMARY_PATH.with_name("exp22_thresholds.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(SUMMARY_PATH, index=False)
    print(f"\n[{time.time()-t0:.0f}s] wrote {OUT_PATH}, {SUMMARY_PATH}, "
          f"{SUMMARY_PATH.with_name('exp22_thresholds.csv')}")


if __name__ == "__main__":
    n_eval = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    run(n_eval)
