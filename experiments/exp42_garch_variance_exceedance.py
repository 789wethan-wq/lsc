"""exp42 -- does a non-CUSUM-on-residuals alarm rule on the SAME GARCH
fit do better than the existing garch_var_cusum, at the cells where
exp32 found GARCH's own conditional-variance path tracks the true
regime even though its CUSUM alarm sits at the FAR floor? (SPEC R7 E,
pre-registered in experiments/CHANGELOG.md 2026-07-27 BEFORE this
script was run.)

Compares garch_var_cusum (CUSUM on GARCH-standardized residuals,
exp15/garch_detector.make_garch_var_cusum_detector) against
garch_variance_exceedance (an exceedance-indicator CUSUM applied
directly to log(sigma2_t), garch_detector.
make_garch_variance_exceedance_detector) -- isolating "GARCH-class
models are the problem" from "this specific wrapper is the problem".

Same 2x2x3 grid as exp15/exp32 (channel in {r, q} x vol_mult in
{1.5, 3.0} x SNR in {0.1, 0.5, 2.0}), phi=0.95, T=500, n_train=125,
n_reps=500, FAR=0.05, same seeds as exp15 (so the existing
garch_var_cusum column can be pulled from exp15's published CSV rather
than recomputed).

Usage: python experiments/exp42_garch_variance_exceedance.py [n_reps]
Output: paper_assets/exp42_garch_variance_exceedance.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from garch_detector import make_garch_variance_exceedance_detector
from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.diagnostics.alarms import calibrate

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "paper_assets" / "exp42_garch_variance_exceedance.csv"
EXP15_PATH = REPO_ROOT / "paper_assets" / "exp15_garch_benchmark.csv"
EXP32_PATH = REPO_ROOT / "paper_assets" / "exp32_garch_mechanism.csv"

PHI, R, T, N_TRAIN = 0.95, 1.0, 500, 125
FAR = 0.05
SEED_CAL, SEED_EVAL = 100_000, 200_000

CHANNELS = ("r", "q")
VOL_MULTS = (1.5, 3.0)
SNRS = (0.1, 0.5, 2.0)


def _exp15_rate(channel: str, snr: float, vol_mult: float, col: str) -> float:
    df = pd.read_csv(EXP15_PATH)
    m = df[(df.channel == channel) & np.isclose(df.snr, snr) & np.isclose(df.vol_mult, vol_mult)]
    return float(m[col].iloc[0]) if len(m) else float("nan")


def _exp32_auc_garch(channel: str, snr: float, vol_mult: float) -> float:
    df = pd.read_csv(EXP32_PATH)
    m = df[(df.channel == channel) & np.isclose(df.snr, snr) & np.isclose(df.vol_mult, vol_mult)]
    return float(m["auc_garch"].iloc[0]) if len(m) else float("nan")


def run_cell(snr: float, channel: str, vol_mult: float, n_reps: int) -> dict:
    q = snr * (1 - PHI**2) * R
    null_dgp = AR1StateDGP(phi=PHI, q=q, r=R)

    exc_fn = make_garch_variance_exceedance_detector(N_TRAIN)
    det_exc = calibrate("garch_variance_exceedance", exc_fn, null_dgp, T, n_reps=n_reps,
                        far=FAR, seed0=SEED_CAL)

    kind = "variance" if channel == "r" else "state_var"
    break_dgp = AR1StateDGP(phi=PHI, q=q, r=R,
                            breaks=[BreakSpec(kind=kind, time_frac=0.5, vol_mult=vol_mult)])
    break_time = break_dgp.breaks[0].time(T)
    paths = [break_dgp.sample(T, seed=SEED_EVAL + i).Y for i in range(n_reps)]

    detect_exc = float(np.mean(
        [det_exc.alarm_time(Y) is not None and det_exc.alarm_time(Y) >= break_time for Y in paths]))

    return dict(
        channel=channel, snr=snr, vol_mult=vol_mult, n_reps=n_reps,
        threshold_exceedance=det_exc.threshold,
        empirical_far_exceedance=float(
            (det_exc.null_max_scores >= det_exc.threshold).mean()),
        detect_garch_cusum=_exp15_rate(channel, snr, vol_mult, "detect_garch"),
        detect_garch_exceedance=detect_exc,
        auc_garch_mechanism=_exp32_auc_garch(channel, snr, vol_mult),
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
                      f"garch_cusum={out['detect_garch_cusum']:.3f} "
                      f"garch_exceedance={out['detect_garch_exceedance']:.3f} "
                      f"(AUC of underlying sigma2_t={out['auc_garch_mechanism']:.3f})",
                      flush=True)
    df = pd.DataFrame(rows).sort_values(["channel", "vol_mult", "snr"]).reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote {OUT_PATH}")
