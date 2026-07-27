"""exp41 -- does a smoothed (two-sided) ARIMA state proxy close any of
the composite-on-ARIMA gap? (SPEC R7 D, pre-registered in
experiments/CHANGELOG.md 2026-07-27 BEFORE this script was run.)

exp20 showed the composite built on ARIMA's one-step-ahead
`fittedvalues` loses decisively to the composite built on the genuine
Kalman filtered state (e.g. 0.226 vs 0.818 at the flagship r-channel
subtle-break, SNR 0.1 cell) -- but 6 of the composite's 11 features
consume `fittedvalues` as a disclosed, not-fully-controlled
state-analog substitution. This asks whether a two-sided,
fixed-interval-smoother state estimate from the SAME frozen ARMA(1,1)
fit -- `lsc.models.SmoothedARIMAModel`, built for this experiment --
closes any of that gap. The smoother is explicitly NOT a causal
detector (it conditions on the whole series, future included); this is
a diagnostic ceiling check, not a proposed deployable rung, the same
oracle-status caveat already given to exp37's break-aware GARCH refit.

Grid: r-channel only, vol_mult=1.5 only (the flagship cell and its two
SNR neighbors -- a targeted follow-up on one already-published gap, not
a new benchmark), SNR in {0.1, 0.5, 2.0}, phi=0.95, T=500, n_train=125,
n_reps=500, FAR=0.05. Reuses exp20's exact seeds/scenario-key
conventions and its published raw / arima-CUSUM / composite-on-Kalman
numbers (not recomputed) so the new smoothed-composite column drops
into the same table.

Usage: python experiments/exp41_smoothed_arima_composite.py [n_reps]
Output: paper_assets/exp41_smoothed_arima_composite.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.diagnostics.alarms import calibrate, empirical_far
from lsc.diagnostics.features import COMPOSITE_V1
from lsc.eval.detectors import make_composite_detector
from lsc.models import ARIMAModel, SmoothedARIMAModel

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "paper_assets" / "exp41_smoothed_arima_composite.csv"

PHI, R, T, N_TRAIN = 0.95, 1.0, 500, 125
FAR = 0.05
SEED_CAL, SEED_EVAL, SEED_FAR_CHECK, SEED_SCALE = 100_000, 200_000, 300_000, 900_000

SNRS = (0.1, 0.5, 2.0)
VOL_MULT = 1.5

# already-published rungs, pulled from exp20's own output -- not recomputed
EXP20_PATH = REPO_ROOT / "paper_assets" / "exp20_composite_on_arima.csv"


def _exp20_rate(snr: float, col: str) -> float:
    df = pd.read_csv(EXP20_PATH)
    m = df[(df.channel == "r") & np.isclose(df.snr, snr) & np.isclose(df.vol_mult, VOL_MULT)]
    return float(m[col].iloc[0]) if len(m) else float("nan")


def run_cell(snr: float, n_reps: int) -> dict:
    q = snr * (1 - PHI**2) * R
    null_dgp = AR1StateDGP(phi=PHI, q=q, r=R)

    fn = make_composite_detector(lambda: SmoothedARIMAModel(), null_dgp, T, N_TRAIN,
                                 n_scale_reps=min(50, n_reps),
                                 scale_seed0=SEED_SCALE, include=COMPOSITE_V1)
    det = calibrate("composite_on_smoothed_arima", fn, null_dgp, T, n_reps=n_reps,
                    far=FAR, seed0=SEED_CAL)
    far_emp = empirical_far(det, null_dgp, T, n_reps=n_reps, seed0=SEED_FAR_CHECK)

    break_dgp = AR1StateDGP(phi=PHI, q=q, r=R,
                             breaks=[BreakSpec(kind="variance", time_frac=0.5, vol_mult=VOL_MULT)])
    break_time = break_dgp.breaks[0].time(T)
    paths = [break_dgp.sample(T, seed=SEED_EVAL + i).Y for i in range(n_reps)]
    detect_smoothed = float(np.mean(
        [det.alarm_time(Y) is not None and det.alarm_time(Y) >= break_time for Y in paths]))

    return dict(
        channel="r", snr=snr, vol_mult=VOL_MULT, n_reps=n_reps,
        threshold_composite_smoothed=det.threshold,
        empirical_far_composite_smoothed=far_emp,
        detect_raw=_exp20_rate(snr, "detect_raw"),
        detect_arima_cusum=_exp20_rate(snr, "detect_arima_cusum"),
        detect_composite_kalman=_exp20_rate(snr, "detect_composite_kalman"),
        detect_composite_arima_onestep=_exp20_rate(snr, "detect_composite_arima"),
        detect_composite_arima_smoothed=detect_smoothed,
    )


def _load_existing() -> pd.DataFrame:
    if OUT_PATH.exists():
        return pd.read_csv(OUT_PATH)
    return pd.DataFrame(columns=["channel", "snr", "vol_mult", "n_reps"])


def _already_done(existing, snr, n_reps):
    if existing.empty:
        return None
    m = existing[np.isclose(existing.snr, snr) & np.isclose(existing.vol_mult, VOL_MULT)
                & (existing.n_reps == n_reps)]
    return m.iloc[0].to_dict() if len(m) else None


if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    t0 = time.time()
    existing = _load_existing()
    rows = []
    for snr in SNRS:
        cached = _already_done(existing, snr, n_reps)
        if cached is not None:
            rows.append(cached)
            print(f"[{time.time()-t0:6.0f}s] SNR={snr}: reused", flush=True)
            continue
        out = run_cell(snr, n_reps)
        rows.append(out)
        print(f"[{time.time()-t0:6.0f}s] SNR={snr}: "
              f"thr={out['threshold_composite_smoothed']:.3f} "
              f"FAR={out['empirical_far_composite_smoothed']:.3f} | "
              f"raw={out['detect_raw']:.3f} "
              f"arima_cusum={out['detect_arima_cusum']:.3f} "
              f"composite_kalman={out['detect_composite_kalman']:.3f} "
              f"composite_arima_onestep={out['detect_composite_arima_onestep']:.3f} "
              f"composite_arima_smoothed={out['detect_composite_arima_smoothed']:.3f}", flush=True)
    df = pd.DataFrame(rows).sort_values(["snr"]).reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote {OUT_PATH}")
