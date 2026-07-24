"""exp24_garch_fresh_far_check.py -- MW4 diagnostic (peer review round
3): does the GARCH(1,1) variance-CUSUM's calibrated threshold
(exp15_garch_benchmark.py) actually deliver ~5% FAR on FRESH null
draws, disjoint from both the calibration seed block (100000+) and the
evaluation seed block (200000+)?

This is the one arm of exp15's grid not yet held to the paper's own
"empirical FAR re-verified on fresh nulls" standard (Contribution 1).
exp15 already established there is no data-snooping between
calibration and evaluation (disjoint seeds, different DGP instances,
CHANGELOG 2026-07-23 "external review" entry) -- but "not contaminated
by evaluation draws" is a different, weaker claim than "the calibrated
threshold delivers 5% FAR out of sample," which matters more for GARCH
than for raw/ARIMA given its heavier-tailed, order-statistic threshold
(Sec 8.4 already flags this general risk).

Reuses the project's standing seed layout (experiments/CHANGELOG.md,
2026-07-13): calibration=100000, evaluation=200000, far_check=300000,
feature_scales=900000 -- disjoint by construction. This script
reproduces exp15's exact calibration (same seed0=100000, same n_reps)
so the threshold checked here is the SAME threshold cited in the
paper, then evaluates it on n_reps fresh null draws from seed0=300000.

Usage: python experiments/exp24_garch_fresh_far_check.py [n_reps]
Output: prints per-cell fresh-draw FAR for raw/ARIMA/GARCH; writes
paper_assets/exp24_garch_fresh_far_check.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lsc.dgp import AR1StateDGP
from lsc.diagnostics.alarms import calibrate, empirical_far
from lsc.eval.detectors import make_arima_var_cusum_detector, make_raw_var_cusum_detector
from garch_detector import make_garch_var_cusum_detector

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "paper_assets" / "exp24_garch_fresh_far_check.csv"

PHI, R, T, N_TRAIN = 0.95, 1.0, 500, 125
FAR_TARGET = 0.05
SEED_CAL, SEED_FAR_CHECK = 100_000, 300_000

CHANNELS = ("r", "q")  # channel doesn't affect the null DGP or calibration --
# the null has no break -- but exp15 calibrates once per (channel, vol_mult,
# snr) triple even though the null_dgp only depends on snr; reproduce that
# exactly so thresholds match exp15's cited numbers cell-for-cell.
VOL_MULTS = (1.5, 3.0)
SNRS = (0.1, 0.5, 2.0)


def run_cell(snr: float, channel: str, vol_mult: float, n_reps: int) -> dict:
    q = snr * (1 - PHI**2) * R
    null_dgp = AR1StateDGP(phi=PHI, q=q, r=R)

    raw_fn = make_raw_var_cusum_detector(N_TRAIN)
    arima_fn = make_arima_var_cusum_detector(N_TRAIN)
    garch_fn = make_garch_var_cusum_detector(N_TRAIN)

    det_raw = calibrate("raw_var_cusum", raw_fn, null_dgp, T, n_reps=n_reps,
                        far=FAR_TARGET, seed0=SEED_CAL)
    det_arima = calibrate("arima_var_cusum", arima_fn, null_dgp, T, n_reps=n_reps,
                          far=FAR_TARGET, seed0=SEED_CAL)
    det_garch = calibrate("garch_var_cusum", garch_fn, null_dgp, T, n_reps=n_reps,
                          far=FAR_TARGET, seed0=SEED_CAL)

    fresh_far_raw = empirical_far(det_raw, null_dgp, T, n_reps=n_reps, seed0=SEED_FAR_CHECK)
    fresh_far_arima = empirical_far(det_arima, null_dgp, T, n_reps=n_reps, seed0=SEED_FAR_CHECK)
    fresh_far_garch = empirical_far(det_garch, null_dgp, T, n_reps=n_reps, seed0=SEED_FAR_CHECK)

    return dict(
        channel=channel, snr=snr, vol_mult=vol_mult, n_reps=n_reps,
        threshold_garch=det_garch.threshold,
        tautological_far_garch=float((det_garch.null_max_scores >= det_garch.threshold).mean()),
        fresh_far_raw=fresh_far_raw, fresh_far_arima=fresh_far_arima,
        fresh_far_garch=fresh_far_garch,
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
    # calibration and the fresh-draw FAR check depend only on snr (the null
    # DGP has no break -- channel/vol_mult only matter for detection rates,
    # which this script does not compute), so compute once per SNR and
    # replicate the row across the 4 (channel, vol_mult) combinations to
    # match exp15's 12-row grid shape for the paper table.
    per_snr_cache = {}
    for snr in SNRS:
        cached = _already_done(existing, CHANNELS[0], snr, VOL_MULTS[0], n_reps)
        if cached is not None:
            per_snr_cache[snr] = cached
            print(f"[{time.time()-t0:6.0f}s] SNR={snr}: reused", flush=True)
            continue
        out = run_cell(snr, CHANNELS[0], VOL_MULTS[0], n_reps)
        per_snr_cache[snr] = out
        print(f"[{time.time()-t0:6.0f}s] SNR={snr}: thr_garch={out['threshold_garch']:.3f} "
              f"fresh_far raw={out['fresh_far_raw']:.3f} "
              f"arima={out['fresh_far_arima']:.3f} "
              f"garch={out['fresh_far_garch']:.3f}", flush=True)

    for channel in CHANNELS:
        for vol_mult in VOL_MULTS:
            for snr in SNRS:
                base = dict(per_snr_cache[snr])
                base["channel"], base["vol_mult"] = channel, vol_mult
                rows.append(base)
    df = pd.DataFrame(rows).sort_values(["channel", "vol_mult", "snr"]).reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote {OUT_PATH}")
    print("\nGARCH fresh-draw FAR summary (target 5%):")
    print(df[["channel", "snr", "vol_mult", "threshold_garch", "fresh_far_garch"]]
         .to_string(index=False))
