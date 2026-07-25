"""exp28 -- known-parameter variance ladder at phi=0.99 (SPEC R2 M1,
pre-registered in experiments/CHANGELOG.md 2026-07-24 BEFORE this
script's grid_v9 cell was run): the phi=0.99 counterpart of exp26,
which ran the known-vs-estimated ablation at the paper's body arena
phi=0.95 only. Isolates how much of the phi=0.99 r-channel anomaly
(ARIMA whitening UNDERPERFORMING raw at the subtle x1.5 break, unlike
its flat dominance at phi=0.95 -- see grid_v9_r_phi99_results.csv) is
estimation noise (AIC order selection / MLE difficulty near the unit
root) versus a structural change in what whitening buys you.

Two channels, asymmetric conventions (matching the estimated-rung
grids they're compared against):
  r channel -- phi=0.99 fixed, q = SNR*(1-phi^2)*r for SNR in
               {0.1, 0.5, 2.0} (grid_v9_r_phi99 convention), vol_mult
               in {1.5, 3.0}. Compared against
               paper_assets/grid_v9_r_phi99_results.csv.
  q channel -- phi=0.99 fixed, q=0.04875, r=1.0 FIXED (grid_v8's
               amplification-anchor convention: q,r held fixed, phi
               varied, so a single induced SNR ~ 2.45 at phi=0.99),
               vol_mult in {1.5, 3.0}. Compared against
               paper_assets/grid_v8_phiqbreak_results.csv, arena
               'ar1_phi0.99'.

Usage: python experiments/exp28_known_param_phi99.py [n_reps]
Output: paper_assets/exp28_known_param_phi99.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lsc.benchmarks.variance import (
    known_kalman_var_cusum_score,
    known_raw_var_cusum_score,
)
from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.diagnostics.alarms import calibrate

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "paper_assets" / "exp28_known_param_phi99.csv"

PHI, R, T, N_TRAIN = 0.99, 1.0, 500, 125
FAR = 0.05
SEED_CAL, SEED_EVAL = 100_000, 200_000

VOL_MULTS = (1.5, 3.0)
R_SNRS = (0.1, 0.5, 2.0)
R_Q_BY_SNR = {0.1: 0.00199, 0.5: 0.00995, 2.0: 0.0398}  # grid_v9 convention
Q_CHANNEL_Q = 0.04875  # grid_v8 amplification-anchor convention (q,r fixed)


def _published_r_rate(snr: float, vol_mult: float, method: str) -> float:
    df = pd.read_csv(REPO_ROOT / "paper_assets" / "grid_v9_r_phi99_results.csv")
    scen = "variance_x3" if vol_mult == 3.0 else "variance_x1.5"
    arena = f"ar1_phi99_snr{snr}"
    m = df[(df.arena == arena) & (df.scenario == scen) & (df.method == method)]
    return float(m.detect_rate.iloc[0]) if len(m) else float("nan")


def _published_q_rate(vol_mult: float, method: str) -> float:
    df = pd.read_csv(REPO_ROOT / "paper_assets" / "grid_v8_phiqbreak_results.csv")
    scen = "qvar_x3" if vol_mult == 3.0 else "qvar_x1.5"
    m = df[(df.arena == "ar1_phi0.99") & (df.scenario == scen) & (df.method == method)]
    return float(m.detect_rate.iloc[0]) if len(m) else float("nan")


def run_cell(channel: str, vol_mult: float, n_reps: int, snr: float | None = None) -> dict:
    q = R_Q_BY_SNR[snr] if channel == "r" else Q_CHANNEL_Q
    null_dgp = AR1StateDGP(phi=PHI, q=q, r=R)

    fn_known_raw = lambda Y: known_raw_var_cusum_score(Y, PHI, q, R, N_TRAIN)
    fn_known_kalman = lambda Y: known_kalman_var_cusum_score(Y, PHI, q, R, N_TRAIN)

    det_known_raw = calibrate("known_raw_var_cusum", fn_known_raw, null_dgp, T,
                              n_reps=n_reps, far=FAR, seed0=SEED_CAL)
    det_known_kalman = calibrate("known_kalman_var_cusum", fn_known_kalman, null_dgp, T,
                                 n_reps=n_reps, far=FAR, seed0=SEED_CAL)

    kind = "variance" if channel == "r" else "state_var"
    break_dgp = AR1StateDGP(phi=PHI, q=q, r=R,
                            breaks=[BreakSpec(kind=kind, time_frac=0.5, vol_mult=vol_mult)])
    break_time = break_dgp.breaks[0].time(T)
    paths = [break_dgp.sample(T, seed=SEED_EVAL + i).Y for i in range(n_reps)]

    def detect_rate(det):
        return float(np.mean([det.alarm_time(Y) is not None and det.alarm_time(Y) >= break_time
                              for Y in paths]))

    detect_known_raw = detect_rate(det_known_raw)
    detect_known_kalman = detect_rate(det_known_kalman)
    if channel == "r":
        detect_est_raw = _published_r_rate(snr, vol_mult, "raw_var_cusum")
        detect_est_arima = _published_r_rate(snr, vol_mult, "arima_var_cusum")
    else:
        detect_est_raw = _published_q_rate(vol_mult, "raw_var_cusum")
        detect_est_arima = _published_q_rate(vol_mult, "arima_var_cusum")

    return dict(
        channel=channel, snr=(snr if channel == "r" else 2.45), vol_mult=vol_mult,
        n_reps=n_reps, q=q, r=R, phi=PHI,
        threshold_known_raw=det_known_raw.threshold,
        threshold_known_kalman=det_known_kalman.threshold,
        empirical_far_known_raw=float(
            (det_known_raw.null_max_scores >= det_known_raw.threshold).mean()),
        empirical_far_known_kalman=float(
            (det_known_kalman.null_max_scores >= det_known_kalman.threshold).mean()),
        detect_est_raw=detect_est_raw, detect_known_raw=detect_known_raw,
        gap_raw=detect_known_raw - detect_est_raw,
        detect_est_arima=detect_est_arima, detect_known_kalman=detect_known_kalman,
        gap_kalman=detect_known_kalman - detect_est_arima,
    )


def _load_existing() -> pd.DataFrame:
    if OUT_PATH.exists():
        return pd.read_csv(OUT_PATH)
    return pd.DataFrame(columns=["channel", "snr", "vol_mult", "n_reps"])


def _already_done(existing, channel, snr, vol_mult, n_reps):
    if existing.empty:
        return None
    m = existing[(existing.channel == channel) & np.isclose(existing.snr, snr)
                & np.isclose(existing.vol_mult, vol_mult) & (existing.n_reps == n_reps)]
    return m.iloc[0].to_dict() if len(m) else None


if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    t0 = time.time()
    existing = _load_existing()
    rows = []
    for vol_mult in VOL_MULTS:
        for snr in R_SNRS:
            cached = _already_done(existing, "r", snr, vol_mult, n_reps)
            if cached is not None:
                rows.append(cached)
                print(f"[{time.time()-t0:6.0f}s] channel=r SNR={snr} vol_mult={vol_mult}: reused",
                      flush=True)
                continue
            out = run_cell("r", vol_mult, n_reps, snr=snr)
            rows.append(out)
            print(f"[{time.time()-t0:6.0f}s] channel=r SNR={snr} vol_mult={vol_mult}: "
                  f"raw est={out['detect_est_raw']:.3f} known={out['detect_known_raw']:.3f} "
                  f"(gap {out['gap_raw']:+.3f}) | "
                  f"kalman/arima est={out['detect_est_arima']:.3f} "
                  f"known={out['detect_known_kalman']:.3f} "
                  f"(gap {out['gap_kalman']:+.3f})", flush=True)
    for vol_mult in VOL_MULTS:
        cached = _already_done(existing, "q", 2.45, vol_mult, n_reps)
        if cached is not None:
            rows.append(cached)
            print(f"[{time.time()-t0:6.0f}s] channel=q vol_mult={vol_mult}: reused", flush=True)
            continue
        out = run_cell("q", vol_mult, n_reps)
        rows.append(out)
        print(f"[{time.time()-t0:6.0f}s] channel=q vol_mult={vol_mult}: "
              f"raw est={out['detect_est_raw']:.3f} known={out['detect_known_raw']:.3f} "
              f"(gap {out['gap_raw']:+.3f}) | "
              f"kalman/arima est={out['detect_est_arima']:.3f} "
              f"known={out['detect_known_kalman']:.3f} "
              f"(gap {out['gap_kalman']:+.3f})", flush=True)
    df = pd.DataFrame(rows).sort_values(["channel", "vol_mult", "snr"]).reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote {OUT_PATH}")
