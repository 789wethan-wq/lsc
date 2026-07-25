"""exp29 -- AR(2)+noise core trichotomy check (SPEC R2 M2, pre-registered
in experiments/CHANGELOG.md 2026-07-24 BEFORE this script was run).

The paper's theory (Propositions 1-2, exp07's ARMA(1,1) equivalence) is
derived for AR(1)+noise. This reruns the three comparisons that DEFINE
the paper's trichotomy -- level-shift raw CUSUM vs. innovation CUSUM;
r-break raw_var_cusum vs. arima_var_cusum; q-break raw_var_cusum vs.
arima_var_cusum -- on AR2StateDGP, at one representative SNR (0.5) and
break size (1sigma level / x1.5 r,q) each, under two parameterizations
chosen for real vs. complex characteristic roots:

  real       phi1=1.4, phi2=-0.45  (poles ~ {0.5, 0.9})
  complex    phi1=1.6, phi2=-0.9   (poles complex, modulus ~0.949)

q-channel convention (disclosed, not a default -- CHANGELOG R2 M2):
the q-break scales the SD of the single shock w_t; this is the direct
structural analogue of AR1StateDGP's q-break and side-steps the
otherwise-ambiguous question of which of phi1/phi2 a "state-innovation"
break would touch (a persistence-type break on phi1/phi2 is a
different, unimplemented question).

Seeds disjoint from every published grid: calibration 110000+,
evaluation 210000+ (SPEC R2 M2 pre-registration).

Usage: python experiments/exp29_ar2_trichotomy.py [n_reps]
Output: paper_assets/exp29_ar2_trichotomy.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lsc.benchmarks.variance import arima_var_cusum_score, raw_var_cusum_score
from lsc.benchmarks.changepoint import raw_cusum_score
from lsc.dgp import AR2StateDGP, BreakSpec
from lsc.diagnostics.alarms import calibrate
from lsc.eval.detectors import make_innovation_cusum_detector
from lsc.models import KalmanModel

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "paper_assets" / "exp29_ar2_trichotomy.csv"

T, N_TRAIN = 500, 125
FAR = 0.05
SEED_CAL, SEED_EVAL = 110_000, 210_000
SNR = 0.5
R = 1.0
VOL_MULT = 1.5
LEVEL_MAGNITUDE = 1.0

PARAMETERIZATIONS = {
    "real_roots": dict(phi1=1.4, phi2=-0.45),
    "complex_roots": dict(phi1=1.6, phi2=-0.9),
}


def q_for_snr(phi1: float, phi2: float, r: float, snr: float) -> float:
    """Invert AR2StateDGP.sigma_ref for q at a target SNR = sigma_ref^2 / r,
    the AR(2) counterpart of the AR1 grid convention q = SNR*(1-phi^2)*r."""
    den = (1.0 + phi2) * ((1.0 - phi2) ** 2 - phi1**2)
    return snr * r * den / (1.0 - phi2)


def _detect_rate(det, paths, break_time):
    return float(np.mean([det.alarm_time(Y) is not None and det.alarm_time(Y) >= break_time
                          for Y in paths]))


def run_level(name: str, phi1: float, phi2: float, q: float, n_reps: int) -> dict:
    null_dgp = AR2StateDGP(phi1=phi1, phi2=phi2, q=q, r=R)
    det_raw = calibrate("raw_cusum", lambda Y: raw_cusum_score(Y, n_train=N_TRAIN),
                        null_dgp, T, n_reps=n_reps, far=FAR, seed0=SEED_CAL)
    innov_fn = make_innovation_cusum_detector(lambda: KalmanModel("ar2"), N_TRAIN)
    det_innov = calibrate("innovation_cusum", innov_fn, null_dgp, T,
                          n_reps=n_reps, far=FAR, seed0=SEED_CAL)

    break_dgp = AR2StateDGP(phi1=phi1, phi2=phi2, q=q, r=R,
                            breaks=[BreakSpec("level", 0.5, magnitude=LEVEL_MAGNITUDE)])
    break_time = break_dgp.breaks[0].time(T)
    paths = [break_dgp.sample(T, seed=SEED_EVAL + i).Y for i in range(n_reps)]

    return dict(parameterization=name, channel="level", detector_a="raw_cusum",
               detector_b="innovation_cusum",
               detect_a=_detect_rate(det_raw, paths, break_time),
               detect_b=_detect_rate(det_innov, paths, break_time),
               far_a=float((det_raw.null_max_scores >= det_raw.threshold).mean()),
               far_b=float((det_innov.null_max_scores >= det_innov.threshold).mean()))


def run_var_channel(name: str, channel: str, phi1: float, phi2: float, q: float,
                    n_reps: int) -> dict:
    null_dgp = AR2StateDGP(phi1=phi1, phi2=phi2, q=q, r=R)
    det_raw = calibrate("raw_var_cusum", lambda Y: raw_var_cusum_score(Y, n_train=N_TRAIN),
                        null_dgp, T, n_reps=n_reps, far=FAR, seed0=SEED_CAL)
    det_arima = calibrate("arima_var_cusum", lambda Y: arima_var_cusum_score(Y, n_train=N_TRAIN),
                          null_dgp, T, n_reps=n_reps, far=FAR, seed0=SEED_CAL)

    kind = "variance" if channel == "r" else "state_var"
    break_dgp = AR2StateDGP(phi1=phi1, phi2=phi2, q=q, r=R,
                            breaks=[BreakSpec(kind, 0.5, vol_mult=VOL_MULT)])
    break_time = break_dgp.breaks[0].time(T)
    paths = [break_dgp.sample(T, seed=SEED_EVAL + i).Y for i in range(n_reps)]

    return dict(parameterization=name, channel=channel, detector_a="raw_var_cusum",
               detector_b="arima_var_cusum",
               detect_a=_detect_rate(det_raw, paths, break_time),
               detect_b=_detect_rate(det_arima, paths, break_time),
               far_a=float((det_raw.null_max_scores >= det_raw.threshold).mean()),
               far_b=float((det_arima.null_max_scores >= det_arima.threshold).mean()))


if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    t0 = time.time()
    rows = []
    for name, params in PARAMETERIZATIONS.items():
        q = q_for_snr(params["phi1"], params["phi2"], R, SNR)
        print(f"[{time.time()-t0:6.0f}s] {name}: q={q:.5f} (SNR={SNR})", flush=True)

        out = run_level(name, params["phi1"], params["phi2"], q, n_reps)
        rows.append(out)
        print(f"[{time.time()-t0:6.0f}s] {name} level: raw={out['detect_a']:.3f} "
              f"innov={out['detect_b']:.3f}", flush=True)

        out = run_var_channel(name, "r", params["phi1"], params["phi2"], q, n_reps)
        rows.append(out)
        print(f"[{time.time()-t0:6.0f}s] {name} r-channel: raw_var={out['detect_a']:.3f} "
              f"arima_var={out['detect_b']:.3f}", flush=True)

        out = run_var_channel(name, "q", params["phi1"], params["phi2"], q, n_reps)
        rows.append(out)
        print(f"[{time.time()-t0:6.0f}s] {name} q-channel: raw_var={out['detect_a']:.3f} "
              f"arima_var={out['detect_b']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote {OUT_PATH}")
