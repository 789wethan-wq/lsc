"""exp26 -- known-parameter column for the variance ladder (peer
review round 3, Missing Experiments): exp10 showed a large
estimated-vs-known-parameter gap at the single Table 2 flagship
level-shift cell (0.554 -> 0.970); this extends the same known-vs-
estimated ablation to the variance ladder (Table 3/5's r/q-channel
grid), which exp10 never touched.

Two known-parameter counterparts (lsc.benchmarks.variance):
  known_raw_var_cusum_score    -- standardize by the DGP's analytic
                                   stationary mean/SD instead of the
                                   training-prefix sample mean/SD
  known_kalman_var_cusum_score -- the same three-arm variance CUSUM on
                                   steady-state (true-parameter) Kalman
                                   innovations instead of an MLE-fit
                                   KalmanModel's innovations; the
                                   natural "known" reference point for
                                   arima_var_cusum's estimated whitening

Same grid as exp15/exp24 (channel in {r, q} x vol_mult in {1.5, 3.0} x
SNR in {0.1, 0.5, 2.0}, T=500, n_train=125, n_reps=500, FAR=0.05, same
seeds) so estimated-vs-known gaps line up cell-for-cell against the
already-published raw_var_cusum/arima_var_cusum numbers (Table 3/5,
grid_v4_varbench/grid_v5_qbreak).

Usage: python experiments/exp26_known_param_variance.py [n_reps]
Output: paper_assets/exp26_known_param_variance.csv
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
OUT_PATH = REPO_ROOT / "paper_assets" / "exp26_known_param_variance.csv"

PHI, R, T, N_TRAIN = 0.95, 1.0, 500, 125
FAR = 0.05
SEED_CAL, SEED_EVAL = 100_000, 200_000

CHANNELS = ("r", "q")
VOL_MULTS = (1.5, 3.0)
SNRS = (0.1, 0.5, 2.0)

# published estimated-parameter detect rates for the identical cells
# (Table 3/5), pulled from committed CSVs, not recomputed
PUBLISHED = {
    "r": dict(path=REPO_ROOT / "paper_assets" / "grid_v4_varbench_core_results.csv",
              scen_fmt="variance_x{vm}", arena_fmt="ar1_snr{snr}"),
    "q": dict(path=REPO_ROOT / "paper_assets" / "grid_v5_qbreak_results.csv",
              scen_fmt="qvar_x{vm}", arena_fmt="ar1_snr{snr}"),
}


def _published_rate(channel: str, snr: float, vol_mult: float, method: str) -> float:
    conf = PUBLISHED[channel]
    df = pd.read_csv(conf["path"])
    scen_vm = "3" if vol_mult == 3.0 else "1.5"
    scenario = conf["scen_fmt"].format(vm=scen_vm)
    arena = conf["arena_fmt"].format(snr=snr)
    m = df[(df.arena == arena) & (df.scenario == scenario) & (df.method == method)]
    return float(m.detect_rate.iloc[0]) if len(m) else float("nan")


def run_cell(snr: float, channel: str, vol_mult: float, n_reps: int) -> dict:
    q = snr * (1 - PHI**2) * R
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
    detect_est_raw = _published_rate(channel, snr, vol_mult, "raw_var_cusum")
    detect_est_arima = _published_rate(channel, snr, vol_mult, "arima_var_cusum")

    return dict(
        channel=channel, snr=snr, vol_mult=vol_mult, n_reps=n_reps,
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
    for channel in CHANNELS:
        for vol_mult in VOL_MULTS:
            for snr in SNRS:
                cached = _already_done(existing, channel, snr, vol_mult, n_reps)
                if cached is not None:
                    rows.append(cached)
                    print(f"[{time.time()-t0:6.0f}s] channel={channel} SNR={snr} "
                          f"vol_mult={vol_mult}: reused", flush=True)
                    continue
                out = run_cell(snr, channel, vol_mult, n_reps)
                rows.append(out)
                print(f"[{time.time()-t0:6.0f}s] channel={channel} SNR={snr} "
                      f"vol_mult={vol_mult}: "
                      f"raw est={out['detect_est_raw']:.3f} known={out['detect_known_raw']:.3f} "
                      f"(gap {out['gap_raw']:+.3f}) | "
                      f"kalman/arima est={out['detect_est_arima']:.3f} "
                      f"known={out['detect_known_kalman']:.3f} "
                      f"(gap {out['gap_kalman']:+.3f})", flush=True)
    df = pd.DataFrame(rows).sort_values(["channel", "vol_mult", "snr"]).reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote {OUT_PATH}")
