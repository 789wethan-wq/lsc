"""exp21_composite_innov5.py -- innovation-only reduced composite: do
the five COMPOSITE_V1 features that already substitute cleanly onto
ARIMA residuals reproduce the SAME Kalman-vs-ARIMA composite gap that
exp20 found for the full 11-feature composite?

exp20 (paper_assets/exp20_composite_on_arima.csv) found composite-on-
ARIMA badly underperforms composite-on-Kalman in several r-channel
cells (e.g. r/x1.5/SNR0.1: Kalman 0.818 vs ARIMA 0.226) even though 5
of its 11 features -- break_pressure, variance_pressure,
variance_pressure_slow, variance_quiet, innovation_ac -- act on the
`innovations` series alone, an object exp07's ARMA(1,1) equivalence
says the two models should share exactly (Y is ARMA(1,1) with
theta=rho=phi(1-K), sigma_eps^2=F to machine precision on the null
path). The other 6 features (level_change, slope, acceleration,
instability, persistence, state_shift_pressure) act on the
filtered-state / one-step-forecast, which has no such shared-object
guarantee -- ARIMA's fittedvalues stand in for the Kalman filtered
state only by judgment call (exp20 docstring).

This script isolates the innovation-only 5 as their OWN composite
(`lsc.diagnostics.features.COMPOSITE_INNOV5`) run through the
EXISTING, UNMODIFIED `make_composite_detector` machinery on BOTH
`KalmanModel("ar1")` and `ARIMAModel()`, across the identical 12-cell
grid as exp20: channel in {r, q} x vol_mult in {1.5, 3.0} x SNR in
{0.1, 0.5, 2.0}, phi=0.95, T=500, n_train=125, n_reps=500, FAR=0.05,
same seed layout (calibration 100000+, evaluation 200000+, FAR-check
300000+, feature-scale 900000+).

Two pre-registered readings (report whichever the 12 cells show, not
filtered):
  - composite-on-Kalman-innov5 ~= composite-on-ARIMA-innov5 in a cell
    => that cell's full-composite gap (exp20) is attributable to the
    six filtered-state features -- "the state is informative" beyond
    what its innovations alone carry.
  - a gap already opens at the innovation-only level in a cell => the
    ARIMA one-step-ahead residual is itself a worse input on that
    arena, independent of any state-only features -- destructive
    substitution, not missing state information.

Both composites (Kalman-innov5, ARIMA-innov5) are computed fresh here
-- neither is published anywhere else (exp20's "composite_kalman"
column is the 11-feature COMPOSITE_V1, not COMPOSITE_INNOV5).

Runtime note (CHANGELOG 2026-07-22, exp20 entry): ARIMA-composite
calibration ran ~8.8h wall-clock across exp20's 12 cells, dominated by
a handful of cells with pathologically slow statsmodels ARIMA MLE
convergence during the ~1550 per-cell prefix fits (calibration +
scale-estimation + FAR-check + eval); feature COUNT does not touch
that cost (fits happen once per path, features are computed from the
fit afterward), so the ARIMA side of this script should take
comparably long. Each cell's row is written to CSV immediately after
that cell completes (not just at the end) so a kill mid-run loses at
most the in-flight cell.

Usage: python experiments/exp21_composite_innov5.py [n_reps]
Output: paper_assets/exp21_composite_innov5.csv, one row per cell:
threshold/empirical-FAR/detect-rate for both composite-on-Kalman-
innov5 and composite-on-ARIMA-innov5, plus the exp20 four-column
numbers (raw, ARIMA-CUSUM, full composite-on-Kalman, full
composite-on-ARIMA) joined in for side-by-side reading.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.diagnostics.alarms import calibrate, empirical_far
from lsc.diagnostics.features import COMPOSITE_INNOV5
from lsc.eval.detectors import make_composite_detector
from lsc.models import ARIMAModel, KalmanModel

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "paper_assets" / "exp21_composite_innov5.csv"
EXP20_PATH = REPO_ROOT / "paper_assets" / "exp20_composite_on_arima.csv"

PHI, R, T, N_TRAIN = 0.95, 1.0, 500, 125
FAR = 0.05
SEED_CAL, SEED_EVAL, SEED_FAR_CHECK, SEED_SCALE = 100_000, 200_000, 300_000, 900_000

CHANNELS = ("r", "q")
VOL_MULTS = (1.5, 3.0)
SNRS = (0.1, 0.5, 2.0)


def run_cell(snr: float, channel: str, vol_mult: float, n_reps: int) -> dict:
    q = snr * (1 - PHI**2) * R
    null_dgp = AR1StateDGP(phi=PHI, q=q, r=R)
    kind = "variance" if channel == "r" else "state_var"
    break_dgp = AR1StateDGP(phi=PHI, q=q, r=R,
                             breaks=[BreakSpec(kind=kind, time_frac=0.5, vol_mult=vol_mult)])
    break_time = break_dgp.breaks[0].time(T)
    paths = [break_dgp.sample(T, seed=SEED_EVAL + i).Y for i in range(n_reps)]

    out = dict(channel=channel, snr=snr, vol_mult=vol_mult, n_reps=n_reps)
    for tag, model_factory in (
        ("kalman", lambda: KalmanModel("ar1")),
        ("arima", lambda: ARIMAModel()),
    ):
        fn = make_composite_detector(model_factory, null_dgp, T, N_TRAIN,
                                     n_scale_reps=min(50, n_reps),
                                     scale_seed0=SEED_SCALE, include=COMPOSITE_INNOV5)
        det = calibrate(f"composite_innov5_{tag}", fn, null_dgp, T, n_reps=n_reps,
                        far=FAR, seed0=SEED_CAL)
        far_emp = empirical_far(det, null_dgp, T, n_reps=n_reps, seed0=SEED_FAR_CHECK)
        detect = float(np.mean(
            [det.alarm_time(Y) is not None and det.alarm_time(Y) >= break_time for Y in paths]))
        out[f"threshold_innov5_{tag}"] = det.threshold
        out[f"empirical_far_innov5_{tag}"] = far_emp
        out[f"detect_innov5_{tag}"] = detect
    return out


def _load_existing() -> pd.DataFrame:
    if OUT_PATH.exists():
        return pd.read_csv(OUT_PATH)
    return pd.DataFrame(columns=["channel", "snr", "vol_mult", "n_reps"])


def _already_done(existing: pd.DataFrame, channel: str, snr: float,
                   vol_mult: float, n_reps: int) -> dict | None:
    if existing.empty:
        return None
    m = existing[(existing.channel == channel) & np.isclose(existing.snr, snr)
                 & np.isclose(existing.vol_mult, vol_mult) & (existing.n_reps == n_reps)]
    return m.iloc[0].to_dict() if len(m) else None


def _save(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).sort_values(["channel", "vol_mult", "snr"]).reset_index(drop=True)
    if EXP20_PATH.exists():
        exp20 = pd.read_csv(EXP20_PATH)[
            ["channel", "snr", "vol_mult", "detect_raw", "detect_arima_cusum",
             "detect_composite_kalman", "detect_composite_arima"]
        ].rename(columns={
            "detect_composite_kalman": "detect_composite11_kalman",
            "detect_composite_arima": "detect_composite11_arima",
        })
        df = df.merge(exp20, on=["channel", "snr", "vol_mult"], how="left")
    df.to_csv(OUT_PATH, index=False)
    return df


if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    t0 = time.time()
    existing = _load_existing()
    rows = []
    for channel in CHANNELS:
        for vol_mult in VOL_MULTS:
            for snr in SNRS:
                cached = _already_done(existing, channel, snr, vol_mult, n_reps)
                if cached is not None:
                    rows.append(cached)
                    print(f"[{time.time()-t0:6.0f}s] channel={channel} SNR={snr} "
                          f"vol_mult={vol_mult}: reused from existing "
                          f"{OUT_PATH.name}", flush=True)
                    continue
                out = run_cell(snr, channel, vol_mult, n_reps)
                rows.append(out)
                print(f"[{time.time()-t0:6.0f}s] channel={channel} SNR={snr} "
                      f"vol_mult={vol_mult} n_reps={n_reps}: "
                      f"kalman_innov5={out['detect_innov5_kalman']:.3f} "
                      f"(FAR={out['empirical_far_innov5_kalman']:.3f}) | "
                      f"arima_innov5={out['detect_innov5_arima']:.3f} "
                      f"(FAR={out['empirical_far_innov5_arima']:.3f})", flush=True)
                _save(rows)  # checkpoint after every cell -- ARIMA side is slow (see docstring)
    df = _save(rows)
    print(f"[{time.time()-t0:6.0f}s] wrote {OUT_PATH}")
