"""exp37 -- GARCH oracle break-aware diagnostic (SPEC R5 M2,
pre-registered in experiments/CHANGELOG.md 2026-07-26 BEFORE this
script was run).

Follows directly from exp32 (round 2): GARCH's fitted conditional-
variance path already tracks the true regime about as well as raw/
ARIMA's z^2 baseline in every floor cell, favoring "the CUSUM wrapper
is underpowered" over "GARCH structurally can't see this break." This
asks the natural next question WITHOUT building a new causal detector:
does the CUSUM-wrapper bottleneck persist even with a PERFECTLY-FIT
model, or does it dissolve once fit quality improves?

A literal "fit GARCH separately pre/post the TRUE break and CUSUM the
result" is self-defeating as an alarm mechanism (flagged and confirmed
with the user before this was built): a correctly-refit model's post-
break residuals are z ~ N(0,1) BY CONSTRUCTION -- there is nothing left
to detect once the model is told the truth. So this is run as an
explicit ORACLE / mechanism-diagnostic (same status as the existing
known-parameter columns: exp10, exp26, exp30), NOT a new entry in
exp15's calibrated-FAR table, and the deliverable is not "does the
oracle detect" but a structural comparison:

  z_single  -- exp15/exp32's existing single-regime construction:
               GARCH fit ONCE on Y[:n_train], frozen parameters
               forward-filtered through the whole series (identical
               construction to garch_conditional_variance_path).
  z_oracle  -- z_single unchanged for t < break_time; for t >= break_
               time, a SEPARATE GARCH refit on the TRUE post-break
               segment Y[break_time:] (oracle: uses the ground-truth
               break location a real online method cannot have).

Reports, per cell (T=500, n_train=125, n_reps=500, same 2x2x3 grid as
exp15/exp32, phi=0.95): post-break mean(z^2) under each construction
(z_oracle near 1.0 confirms the self-defeat property directly rather
than assuming it), and whether each construction's per-replicate max
CUSUM score crosses exp15's ALREADY-CALIBRATED threshold (the same
`threshold_garch` used for the real detector) -- demonstrating, rather
than just asserting, that the oracle refit removes the alarm signal
along with the misspecification. The comparison this actually answers:
z_single's post-break mean(z^2) elevation over 1.0 IS the exact
signal the plain-GARCH CUSUM is (weakly) accumulating -- so the size of
that elevation, not the oracle's crossing rate, is the fit-quality
readout: a small elevation would mean little signal was ever there to
extract (wrapper-agnostic); a large elevation the calibrated CUSUM
still fails to convert into alarms would sharpen "the wrapper is the
bottleneck," since perfect knowledge does not add power beyond what
z_single already carries, it just consumes the same signal that
produced z_single's departure from 1.0 in the first place.

Seeds: evaluation-only 400000+, IDENTICAL to exp32's block (this
reproduces exp32's z_single construction bit-for-bit on the same
replicates, a useful self-consistency check, and there is no
downstream paired-decomposition requirement that would demand a fresh
block).

Usage: python experiments/exp37_garch_oracle_break_aware.py [n_reps]
Output: paper_assets/exp37_garch_oracle_break_aware.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from garch_detector import oracle_two_regime_residuals
from lsc.benchmarks.variance import _max_over_arms
from lsc.dgp import AR1StateDGP, BreakSpec

REPO_ROOT = Path(__file__).resolve().parent.parent
A = REPO_ROOT / "paper_assets"
OUT_PATH = A / "exp37_garch_oracle_break_aware.csv"

PHI, R, T, N_TRAIN = 0.95, 1.0, 500, 125
SEED_EVAL = 400_000  # identical to exp32's block

CHANNELS = ("r", "q")
VOL_MULTS = (1.5, 3.0)
SNRS = (0.1, 0.5, 2.0)


def _published_threshold_and_rate(channel: str, snr: float, vol_mult: float) -> tuple[float, float]:
    df = pd.read_csv(A / "exp15_garch_benchmark.csv")
    m = df[(df.channel == channel) & np.isclose(df.snr, snr) & np.isclose(df.vol_mult, vol_mult)]
    row = m.iloc[0]
    return float(row.threshold_garch), float(row.detect_garch)


def run_cell(channel: str, vol_mult: float, snr: float, n_reps: int) -> dict:
    q = snr * (1 - PHI**2) * R
    kind = "variance" if channel == "r" else "state_var"
    break_dgp = AR1StateDGP(phi=PHI, q=q, r=R,
                            breaks=[BreakSpec(kind=kind, time_frac=0.5, vol_mult=vol_mult)])
    break_time = break_dgp.breaks[0].time(T)
    threshold_garch, published_detect = _published_threshold_and_rate(channel, snr, vol_mult)

    post_z2_single, post_z2_oracle = [], []
    single_crosses, oracle_crosses = [], []
    for i in range(n_reps):
        Y = break_dgp.sample(T, seed=SEED_EVAL + i).Y
        z_single, z_oracle = oracle_two_regime_residuals(Y, N_TRAIN, break_time)

        post_z2_single.append(np.mean(z_single[break_time:] ** 2))
        post_z2_oracle.append(np.mean(z_oracle[break_time:] ** 2))

        score_single = _max_over_arms(z_single, N_TRAIN)
        score_oracle = _max_over_arms(z_oracle, N_TRAIN)
        single_crosses.append(bool(np.nanmax(score_single[N_TRAIN:]) >= threshold_garch))
        oracle_crosses.append(bool(np.nanmax(score_oracle[N_TRAIN:]) >= threshold_garch))

    return dict(
        channel=channel, snr=snr, vol_mult=vol_mult, n_reps=n_reps,
        mean_post_break_z2_single=float(np.mean(post_z2_single)),
        mean_post_break_z2_oracle=float(np.mean(post_z2_oracle)),
        z2_elevation_single=float(np.mean(post_z2_single) - 1.0),
        z2_elevation_oracle=float(np.mean(post_z2_oracle) - 1.0),
        cross_rate_single_at_garch_threshold=float(np.mean(single_crosses)),
        cross_rate_oracle_at_garch_threshold=float(np.mean(oracle_crosses)),
        published_detect_garch=published_detect,
    )


if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    t0 = time.time()
    rows = []
    for channel in CHANNELS:
        for vol_mult in VOL_MULTS:
            for snr in SNRS:
                out = run_cell(channel, vol_mult, snr, n_reps)
                rows.append(out)
                print(f"[{time.time()-t0:6.0f}s] {channel} x{vol_mult} SNR{snr}: "
                      f"z2_single={out['mean_post_break_z2_single']:.3f} "
                      f"(elev {out['z2_elevation_single']:+.3f}) "
                      f"z2_oracle={out['mean_post_break_z2_oracle']:.3f} "
                      f"(elev {out['z2_elevation_oracle']:+.3f}) | "
                      f"cross_single={out['cross_rate_single_at_garch_threshold']:.3f} "
                      f"(published_detect={out['published_detect_garch']:.3f}) "
                      f"cross_oracle={out['cross_rate_oracle_at_garch_threshold']:.3f}",
                      flush=True)
    df = pd.DataFrame(rows).sort_values(["channel", "vol_mult", "snr"]).reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote {OUT_PATH}")
