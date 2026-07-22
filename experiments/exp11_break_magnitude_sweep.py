"""exp11 -- dense break-magnitude sweep (referee-requested, external
review this round): the paper's headline level and r-channel results
(§4, §5) are demonstrated at a handful of discrete magnitudes (0.5/1/3
sigma_ref for levels; x1.5/x3 for observation-noise variance). This
runs a dense continuum instead, using the repo's ACTUAL modules and
ESTIMATED (training-prefix MLE) parameters -- not a from-scratch,
known-parameter reimplementation -- to check whether raw's dominance
(levels) and the raw/ARIMA ordering (r-channel, Table 3) hold
throughout, not just at the reported points.

Level channel (Sec 4): make_raw_cusum_detector vs
make_innovation_cusum_detector (lsc.eval.detectors), phi=0.95, T=500,
n_train=125 (grid_v1 convention), across a 21-point magnitude grid
(0.0-4.0 sigma_ref in steps of 0.2), at SNR in {0.1, 0.5, 2.0}. Also
reports the theoretical mu_inf (Proposition 1, TRUE arena params, since
it is a property of the DGP the estimated filter is approximating, not
an estimated quantity itself) for the knife-edge cross-reference.

r-channel (Sec 5): make_raw_var_cusum_detector vs
make_arima_var_cusum_detector (lsc.eval.detectors / lsc.benchmarks.variance),
same arena, 21-point vol_mult grid (1.0-3.5 in steps of 0.125), same
three SNRs.

n_reps=500 and seed0=100000 (calibration) / 200000 (evaluation) match
configs/grid_v1.yaml and grid_v4_varbench_core.yaml exactly -- every
method within an arena calibrates against the SAME null draws, every
magnitude point evaluates against the SAME break-path draws (only the
break's own parameters differ), matching lsc.eval.runner's convention.
This is a correction from an earlier run of this script that used
n_reps=250 and separate seed offsets per detector, which produced
checkpoint mismatches against Tables 2/3 traced to calibration-
threshold sampling variability at the smaller n.

Usage: python experiments/exp11_break_magnitude_sweep.py [n_reps]
Output: paper_assets/exp11_level_sweep.csv,
        paper_assets/exp11_rchannel_sweep.csv
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.diagnostics.alarms import calibrate
from lsc.eval.detectors import (
    make_arima_var_cusum_detector,
    make_innovation_cusum_detector,
    make_raw_cusum_detector,
    make_raw_var_cusum_detector,
)
from lsc.models import KalmanModel
from lsc.theory import mu_infinity, riccati_steady_state

PHI, R = 0.95, 1.0
T, TRAIN_FRAC = 500, 0.25
N_TRAIN = int(round(TRAIN_FRAC * T))
FAR = 0.05
# matches lsc.eval.runner / configs/grid_v1.yaml, grid_v4_varbench_core.yaml
# exactly: every method within an arena is calibrated against the SAME
# null draws (seed0=100000+i), and every scenario is evaluated against
# the SAME break-path draws (seed0=200000+i) -- not separate seed
# blocks per detector. This is required for the dense-sweep checkpoints
# (ratio=3.0 level; vol_mult=1.5 r-channel) to reproduce Table 2/3
# exactly rather than approximately.
SEED_CAL, SEED_EVAL = 100_000, 200_000

LEVEL_RATIOS = np.round(np.arange(0.0, 4.01, 0.2), 2)       # 21 points
VOL_MULTS = np.round(np.arange(1.0, 3.51, 0.125), 3)        # 21 points


def level_sweep(n_reps: int, snrs=(0.1, 0.5, 2.0)) -> pd.DataFrame:
    rows = []
    for snr in snrs:
        q = snr * (1 - PHI**2)
        _, K, F = riccati_steady_state(PHI, q, R)
        sigma_ref = np.sqrt(q / (1 - PHI**2))
        null_dgp = AR1StateDGP(phi=PHI, q=q, r=R)

        raw_fn = make_raw_cusum_detector(N_TRAIN)
        innov_fn = make_innovation_cusum_detector(lambda: KalmanModel("ar1"), N_TRAIN)
        det_raw = calibrate("raw_cusum", raw_fn, null_dgp, T, n_reps=n_reps,
                            far=FAR, seed0=SEED_CAL)
        det_innov = calibrate("lsc_kalman_cusum", innov_fn, null_dgp, T,
                              n_reps=n_reps, far=FAR, seed0=SEED_CAL)

        for ratio in LEVEL_RATIOS:
            delta = ratio * sigma_ref
            mu_inf = mu_infinity(delta, PHI, K, F)
            break_dgp = AR1StateDGP(phi=PHI, q=q, r=R, breaks=[
                BreakSpec(kind="level", time_frac=0.5, magnitude=ratio)])
            break_time = break_dgp.breaks[0].time(T)
            paths = [break_dgp.sample(T, seed=SEED_EVAL + i).Y for i in range(n_reps)]
            d_raw = np.mean([det_raw.alarm_time(Y) is not None
                             and det_raw.alarm_time(Y) >= break_time for Y in paths])
            d_innov = np.mean([det_innov.alarm_time(Y) is not None
                               and det_innov.alarm_time(Y) >= break_time for Y in paths])
            rows.append(dict(channel="level", snr=float(snr), ratio=ratio,
                             mu_inf=round(mu_inf, 4),
                             det_raw=float(d_raw), det_innov=float(d_innov)))
        print(f"[level] snr={snr} done", flush=True)
    return pd.DataFrame(rows)


def rchannel_sweep(n_reps: int, snrs=(0.1, 0.5, 2.0)) -> pd.DataFrame:
    rows = []
    for snr in snrs:
        q = snr * (1 - PHI**2)
        null_dgp = AR1StateDGP(phi=PHI, q=q, r=R)

        raw_fn = make_raw_var_cusum_detector(N_TRAIN)
        arima_fn = make_arima_var_cusum_detector(N_TRAIN)
        det_raw = calibrate("raw_var_cusum", raw_fn, null_dgp, T, n_reps=n_reps,
                            far=FAR, seed0=SEED_CAL)
        det_arima = calibrate("arima_var_cusum", arima_fn, null_dgp, T,
                              n_reps=n_reps, far=FAR, seed0=SEED_CAL)

        for mult in VOL_MULTS:
            break_dgp = AR1StateDGP(phi=PHI, q=q, r=R, breaks=[
                BreakSpec(kind="variance", time_frac=0.5, vol_mult=float(mult))])
            break_time = break_dgp.breaks[0].time(T)
            paths = [break_dgp.sample(T, seed=SEED_EVAL + i).Y
                    for i in range(n_reps)]
            d_raw = np.mean([det_raw.alarm_time(Y) is not None
                             and det_raw.alarm_time(Y) >= break_time for Y in paths])
            d_arima = np.mean([det_arima.alarm_time(Y) is not None
                               and det_arima.alarm_time(Y) >= break_time for Y in paths])
            rows.append(dict(channel="r", snr=snr, vol_mult=mult,
                             det_raw=float(d_raw), det_arima=float(d_arima)))
        print(f"[r-channel] snr={snr} done", flush=True)
    return pd.DataFrame(rows)


def main(n_reps: int = 500) -> None:
    t0 = time.time()
    lvl = level_sweep(n_reps)
    lvl.to_csv("paper_assets/exp11_level_sweep.csv", index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote paper_assets/exp11_level_sweep.csv")

    rch = rchannel_sweep(n_reps)
    rch.to_csv("paper_assets/exp11_rchannel_sweep.csv", index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote paper_assets/exp11_rchannel_sweep.csv")

    print("\n=== level channel: raw-dominance violations "
          "(det_raw < det_innov - 0.03) ===")
    viol = lvl[lvl.det_raw < lvl.det_innov - 0.03]
    print(viol.to_string(index=False) if len(viol) else "none")

    print("\n=== level channel: knife-edge crossing (mu_inf=0.5) vs "
          "empirical det_innov>=0.5 crossing ===")
    for snr, g in lvl.groupby("snr"):
        g = g.sort_values("ratio")
        theo = np.interp(0.5, g.mu_inf, g.ratio) if g.mu_inf.max() >= 0.5 else float("nan")
        emp_rows = g[g.det_innov >= 0.5]
        emp = float(emp_rows.ratio.iloc[0]) if len(emp_rows) else float("nan")
        print(f"  SNR={snr}: mu_inf=0.5 at ratio~{theo:.2f}, "
              f"det_innov>=0.5 first at ratio~{emp}")

    print("\n=== r-channel: raw vs ARIMA crossover (where does the "
          "ordering flip?) ===")
    for snr, g in rch.groupby("snr"):
        g = g.sort_values("vol_mult")
        raw_wins = g[g.det_raw > g.det_arima + 0.03]
        arima_wins = g[g.det_arima > g.det_raw + 0.03]
        print(f"  SNR={snr}: raw>ARIMA at vol_mult="
              f"{raw_wins.vol_mult.tolist() if len(raw_wins) else 'never'}; "
              f"ARIMA>raw at vol_mult="
              f"{arima_wins.vol_mult.tolist() if len(arima_wins) else 'never'}")

    print(f"\n[{time.time()-t0:6.0f}s] done")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
