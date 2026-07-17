"""exp08 -- PELT (offline changepoint) as an actual competing-method
comparison, not just dismissed in related work (publishability review,
CHANGELOG 2026-07-16 P2 entry).

PELT (ruptures, l2 cost, applied to Y standardized by training-prefix
mean/std -- the same standardization as the raw_cusum benchmark, so
this sits on the "raw" rung of the whitening ladder) is calibrated to
a target false-alarm rate on null AR(1) paths by bisecting its penalty
parameter (higher pen -> fewer reported breakpoints -> lower FAR),
mirroring the threshold-calibration protocol used for every causal
detector (SPEC Sec 4.2) so the comparison is FAR-matched.

Evaluation is OFFLINE LOCALIZATION, not delay: does PELT report >= 1
breakpoint within WINDOW obs of the true break, given the FULL sample.
This is explicitly not comparable to the causal detectors' delay
numbers (PELT sees future data) -- SPEC Sec 4.1 excludes PELT from
delay tables, and this script does not report a delay; it reports
whether an off-the-shelf offline method localizes the break at all, at
a FAR-matched operating point, as a direct check on the "why not just
use PELT" question the related-work section otherwise only asserts.

Both the training prefix and the FAR-check/evaluation segment
generation reuse the exact arena parameters (phi, q, r) and break
scenarios of grid_v1's AR(1) core slice, so localize_rate is directly
comparable to that table's raw_cusum detect_rate.

Usage: python experiments/exp08_pelt_benchmark.py [n_reps]
Outputs: paper_assets/exp08_pelt_results.csv,
         paper_assets/exp08_pelt_far_calibration.csv
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from lsc.benchmarks.changepoint import pelt_breakpoints
from lsc.dgp import AR1StateDGP, BreakSpec

T = 500
N_TRAIN = 125
FAR_TARGET = 0.05
WINDOW = 25  # obs tolerance for "found the break" localization
SEEDS = dict(calibration=100_000, evaluation=200_000)

ARENAS = {
    "ar1_snr0.1": dict(phi=0.95, q=0.00975, r=1.0),
    "ar1_snr0.5": dict(phi=0.95, q=0.04875, r=1.0),
    "ar1_snr2.0": dict(phi=0.95, q=0.195, r=1.0),
}

SCENARIOS = {
    "level_0.5s": dict(kind="level", time_frac=0.5, magnitude=0.5),
    "level_1s": dict(kind="level", time_frac=0.5, magnitude=1.0),
    "level_3s": dict(kind="level", time_frac=0.5, magnitude=3.0),
    "variance_x1.5": dict(kind="variance", time_frac=0.5, vol_mult=1.5),
    "variance_x3": dict(kind="variance", time_frac=0.5, vol_mult=3.0),
}


def _standardized_posttrain(Y: np.ndarray) -> np.ndarray:
    """Same standardization as raw_cusum_score; restricted to the
    post-training segment so PELT gets no more information than the
    causal detectors' training split allows."""
    mu = Y[:N_TRAIN].mean()
    sd = max(Y[:N_TRAIN].std(ddof=1), 1e-12)
    return (Y[N_TRAIN:] - mu) / sd


def pelt_fires(Y: np.ndarray, pen: float) -> bool:
    seg = _standardized_posttrain(Y)
    return len(pelt_breakpoints(seg, pen=pen)) > 0


def pelt_localizes(Y: np.ndarray, pen: float, break_time: int) -> bool:
    seg = _standardized_posttrain(Y)
    b_rel = break_time - N_TRAIN
    return any(abs(b - b_rel) <= WINDOW for b in pelt_breakpoints(seg, pen=pen))


def calibrate_pen(null_dgp, n_reps: int, far_target: float, seed0: int,
                  lo: float = 0.5, hi: float = 500.0,
                  n_bisect: int = 12) -> tuple[float, float]:
    """Bisect pen so P(pelt_fires | null) ~= far_target (higher pen ->
    fewer breakpoints -> lower FAR, so the search direction mirrors a
    monotone-decreasing FAR(pen))."""
    Ys = [null_dgp.sample(T, seed=seed0 + i).Y for i in range(n_reps)]
    if pelt_fires_rate(Ys, hi) > far_target:
        raise RuntimeError(f"hi={hi} pen insufficient to reach FAR "
                           f"{far_target}; widen the bisection bracket")

    def far_at(pen: float) -> float:
        return float(np.mean([pelt_fires(Y, pen) for Y in Ys]))

    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if far_at(mid) > far_target:
            lo = mid
        else:
            hi = mid
    pen = 0.5 * (lo + hi)
    return pen, far_at(pen)


def pelt_fires_rate(Ys: list[np.ndarray], pen: float) -> float:
    return float(np.mean([pelt_fires(Y, pen) for Y in Ys]))


def main(n_reps: int = 300) -> None:
    t0 = time.time()
    far_rows, rows = [], []
    for arena_name, arena_p in ARENAS.items():
        null_dgp = AR1StateDGP(**arena_p, breaks=[])
        pen, far_emp = calibrate_pen(null_dgp, n_reps, FAR_TARGET,
                                     SEEDS["calibration"])
        far_rows.append(dict(arena=arena_name, pen=pen,
                             far_target=FAR_TARGET, far_empirical=far_emp))
        print(f"[{time.time()-t0:6.0f}s] calibrated PELT {arena_name}: "
              f"pen={pen:.2f} FAR={far_emp:.3f}", flush=True)

        for scen_name, spec_kwargs in SCENARIOS.items():
            spec = BreakSpec(**spec_kwargs)
            dgp = AR1StateDGP(**arena_p, breaks=[spec])
            b_time = spec.time(T)
            hits = [pelt_localizes(dgp.sample(T, seed=SEEDS["evaluation"] + i).Y,
                                   pen, b_time)
                    for i in range(n_reps)]
            rate = float(np.mean(hits))
            se = float(np.std(hits, ddof=1) / np.sqrt(len(hits)))
            rows.append(dict(arena=arena_name, scenario=scen_name,
                             localize_rate=rate, localize_rate_se=se,
                             n=n_reps, window=WINDOW))
            print(f"[{time.time()-t0:6.0f}s] {arena_name} {scen_name}: "
                  f"localize={rate:.2f}", flush=True)

    pd.DataFrame(far_rows).to_csv(
        "paper_assets/exp08_pelt_far_calibration.csv", index=False)
    pd.DataFrame(rows).to_csv(
        "paper_assets/exp08_pelt_results.csv", index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote paper_assets/exp08_pelt_*",
          flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
