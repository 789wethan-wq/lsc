"""exp43 -- systematic paired SEs for Table 2b's known-vs-estimated
detection-rate gaps (SPEC R7 F, pre-registered in
experiments/CHANGELOG.md 2026-07-27 BEFORE this script was run).

exp40 covered Tables 3/3b/3c/4 (raw-vs-ARIMA) but not Table 2b
(known-vs-estimated, exp26's 12 cells), where some 0.01-0.02 gaps are
called "within MC noise" using only an independence-bound SE. Reuses
exp26's and exp15's exact seeds/construction (same null_dgp, same
SEED_CAL/SEED_EVAL, same calibrate() calls) so BOTH the known rung
(known_raw_var_cusum / known_kalman_var_cusum) and the estimated rung
(raw_var_cusum / arima_var_cusum) are reconstructed on the SAME
per-replicate evaluation draws -- verified against each rung's already-
published aggregate before trusting the pairing, exactly as exp40's
methodology requires.

Grid: channel in {r, q} x vol_mult in {1.5, 3.0} x SNR in
{0.1, 0.5, 2.0}, phi=0.95, T=500, n_train=125, n_reps=500, FAR=0.05 --
the same 12 cells as Table 2b / exp26.

Usage: python experiments/exp43_table2b_paired_se.py [n_reps]
Output: paper_assets/exp43_table2b_paired_se.csv (one row per cell:
        channel, snr, vol_mult, gap_raw, gap_kalman, se_paired_raw,
        se_independence_raw, se_paired_kalman, se_independence_kalman,
        tightens_under_pairing_raw, tightens_under_pairing_kalman)
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
from lsc.eval.detectors import make_arima_var_cusum_detector, make_raw_var_cusum_detector

REPO_ROOT = Path(__file__).resolve().parent.parent
A = REPO_ROOT / "paper_assets"
OUT_PATH = A / "exp43_table2b_paired_se.csv"
EXP26_PATH = A / "exp26_known_param_variance.csv"

PHI, R, T, N_TRAIN = 0.95, 1.0, 500, 125
FAR = 0.05
SEED_CAL, SEED_EVAL = 100_000, 200_000

CHANNELS = ("r", "q")
VOL_MULTS = (1.5, 3.0)
SNRS = (0.1, 0.5, 2.0)

PUBLISHED = {
    "r": dict(path=REPO_ROOT / "paper_assets" / "grid_v4_varbench_core_results.csv",
              scen_fmt="variance_x{vm}", arena_fmt="ar1_snr{snr}"),
    "q": dict(path=REPO_ROOT / "paper_assets" / "grid_v5_qbreak_results.csv",
              scen_fmt="qvar_x{vm}", arena_fmt="ar1_snr{snr}"),
}


def _published_estimated(channel: str, snr: float, vol_mult: float, method: str) -> float:
    conf = PUBLISHED[channel]
    df = pd.read_csv(conf["path"])
    scen_vm = "3" if vol_mult == 3.0 else "1.5"
    scenario = conf["scen_fmt"].format(vm=scen_vm)
    arena = conf["arena_fmt"].format(snr=snr)
    m = df[(df.arena == arena) & (df.scenario == scenario) & (df.method == method)]
    return float(m.detect_rate.iloc[0]) if len(m) else float("nan")


def _published_known(channel: str, snr: float, vol_mult: float, col: str) -> float:
    df = pd.read_csv(EXP26_PATH)
    m = df[(df.channel == channel) & np.isclose(df.snr, snr) & np.isclose(df.vol_mult, vol_mult)]
    return float(m[col].iloc[0]) if len(m) else float("nan")


def reconstruct_cell(channel: str, snr: float, vol_mult: float, n_reps: int) -> dict:
    q = snr * (1 - PHI**2) * R
    null_dgp = AR1StateDGP(phi=PHI, q=q, r=R)

    fn_known_raw = lambda Y: known_raw_var_cusum_score(Y, PHI, q, R, N_TRAIN)
    fn_known_kalman = lambda Y: known_kalman_var_cusum_score(Y, PHI, q, R, N_TRAIN)
    fn_est_raw = make_raw_var_cusum_detector(N_TRAIN)
    fn_est_arima = make_arima_var_cusum_detector(N_TRAIN)

    det = {}
    det["known_raw"] = calibrate("known_raw_var_cusum", fn_known_raw, null_dgp, T,
                                 n_reps=n_reps, far=FAR, seed0=SEED_CAL)
    det["known_kalman"] = calibrate("known_kalman_var_cusum", fn_known_kalman, null_dgp, T,
                                    n_reps=n_reps, far=FAR, seed0=SEED_CAL)
    det["est_raw"] = calibrate("raw_var_cusum", fn_est_raw, null_dgp, T,
                               n_reps=n_reps, far=FAR, seed0=SEED_CAL)
    det["est_arima"] = calibrate("arima_var_cusum", fn_est_arima, null_dgp, T,
                                 n_reps=n_reps, far=FAR, seed0=SEED_CAL)

    kind = "variance" if channel == "r" else "state_var"
    break_dgp = AR1StateDGP(phi=PHI, q=q, r=R,
                            breaks=[BreakSpec(kind=kind, time_frac=0.5, vol_mult=vol_mult)])
    break_time = break_dgp.breaks[0].time(T)
    paths = [break_dgp.sample(T, seed=SEED_EVAL + i).Y for i in range(n_reps)]

    detected = {}
    for name, d in det.items():
        arr = np.empty(n_reps, dtype=bool)
        for i, Y in enumerate(paths):
            at = d.alarm_time(Y)
            arr[i] = at is not None and at >= break_time
        detected[name] = arr

    p_known_raw = float(detected["known_raw"].mean())
    p_known_kalman = float(detected["known_kalman"].mean())
    p_est_raw = float(detected["est_raw"].mean())
    p_est_arima = float(detected["est_arima"].mean())

    pub_known_raw = _published_known(channel, snr, vol_mult, "detect_known_raw")
    pub_known_kalman = _published_known(channel, snr, vol_mult, "detect_known_kalman")
    pub_est_raw = _published_estimated(channel, snr, vol_mult, "raw_var_cusum")
    pub_est_arima = _published_estimated(channel, snr, vol_mult, "arima_var_cusum")

    d_raw = detected["known_raw"].astype(float) - detected["est_raw"].astype(float)
    d_kalman = detected["known_kalman"].astype(float) - detected["est_arima"].astype(float)
    se_paired_raw = float(d_raw.std(ddof=1) / np.sqrt(n_reps))
    se_paired_kalman = float(d_kalman.std(ddof=1) / np.sqrt(n_reps))
    se_indep_raw = float(np.sqrt(p_known_raw * (1 - p_known_raw) / n_reps
                                 + p_est_raw * (1 - p_est_raw) / n_reps))
    se_indep_kalman = float(np.sqrt(p_known_kalman * (1 - p_known_kalman) / n_reps
                                    + p_est_arima * (1 - p_est_arima) / n_reps))

    return dict(
        channel=channel, snr=snr, vol_mult=vol_mult, n_reps=n_reps,
        detect_known_raw=p_known_raw, detect_est_raw=p_est_raw, gap_raw=p_known_raw - p_est_raw,
        reproduced_known_raw=bool(np.isclose(p_known_raw, pub_known_raw, atol=1e-9)),
        reproduced_est_raw=bool(np.isclose(p_est_raw, pub_est_raw, atol=1e-9)),
        se_paired_raw=se_paired_raw, se_independence_raw=se_indep_raw,
        tightens_under_pairing_raw=bool(se_paired_raw < se_indep_raw),
        detect_known_kalman=p_known_kalman, detect_est_arima=p_est_arima,
        gap_kalman=p_known_kalman - p_est_arima,
        reproduced_known_kalman=bool(np.isclose(p_known_kalman, pub_known_kalman, atol=1e-9)),
        reproduced_est_arima=bool(np.isclose(p_est_arima, pub_est_arima, atol=1e-9)),
        se_paired_kalman=se_paired_kalman, se_independence_kalman=se_indep_kalman,
        tightens_under_pairing_kalman=bool(se_paired_kalman < se_indep_kalman),
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
                out = reconstruct_cell(channel, snr, vol_mult, n_reps)
                rows.append(out)
                print(f"[{time.time()-t0:6.0f}s] channel={channel} SNR={snr} "
                      f"vol_mult={vol_mult}: gap_raw={out['gap_raw']:+.3f} "
                      f"SE_paired={out['se_paired_raw']:.4f} "
                      f"SE_indep={out['se_independence_raw']:.4f} | "
                      f"gap_kalman={out['gap_kalman']:+.3f} "
                      f"SE_paired={out['se_paired_kalman']:.4f} "
                      f"SE_indep={out['se_independence_kalman']:.4f} "
                      f"reproduced=({out['reproduced_known_raw']},{out['reproduced_est_raw']},"
                      f"{out['reproduced_known_kalman']},{out['reproduced_est_arima']})",
                      flush=True)
    df = pd.DataFrame(rows).sort_values(["channel", "vol_mult", "snr"]).reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    n_not_tighten = int((~df.tightens_under_pairing_raw).sum()
                        + (~df.tightens_under_pairing_kalman).sum())
    n_not_reproduced = int((~(df.reproduced_known_raw & df.reproduced_est_raw
                             & df.reproduced_known_kalman & df.reproduced_est_arima)).sum())
    print(f"\n[{time.time()-t0:6.0f}s] {len(df)} cells total; "
          f"{n_not_tighten}/{2*len(df)} rung-pairs did NOT tighten under pairing; "
          f"{n_not_reproduced} cells had a reconstruction mismatch")
    print(f"wrote {OUT_PATH}")
