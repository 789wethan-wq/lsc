"""exp25 -- ICSS (Inclan & Tiao 1994) as the variance-changepoint
counterpart to exp08's PELT (mean-shift) benchmark.

Peer review round 3 (Missing Experiments): the paper's §8.5 already
notes PELT is the wrong tool for variance breaks -- ICSS is the
benchmark conspicuously missing from a paper about variance-break
detection. Mirrors exp08_pelt_benchmark.py's design exactly: OFFLINE
LOCALIZATION on the full (standardized, post-training) sample, not a
causal delay comparison (SPEC §4.1 excludes offline methods from
delay tables, same treatment PELT gets); calibrated by simulation to a
target FAR via bisection on the ICSS D-statistic's threshold (`crit`),
mirroring PELT's `pen` bisection; same arenas, same seeds, same
localization WINDOW, so localize_rate is directly comparable to
exp08's PELT numbers and to Table 3/5's causal raw_var_cusum rates.
Restricted to the variance scenarios only (ICSS has no claim on level
shifts) -- see lsc.benchmarks.changepoint.icss_breakpoints for the
algorithm.

Usage: python experiments/exp25_icss_benchmark.py [n_reps]
Outputs: paper_assets/exp25_icss_results.csv,
         paper_assets/exp25_icss_far_calibration.csv
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from lsc.benchmarks.changepoint import icss_breakpoints
from lsc.dgp import AR1StateDGP, BreakSpec

T = 500
N_TRAIN = 125
FAR_TARGET = 0.05
WINDOW = 25  # obs tolerance for "found the break" localization -- matches exp08
SEEDS = dict(calibration=100_000, evaluation=200_000)

ARENAS = {
    "ar1_snr0.1": dict(phi=0.95, q=0.00975, r=1.0),
    "ar1_snr0.5": dict(phi=0.95, q=0.04875, r=1.0),
    "ar1_snr2.0": dict(phi=0.95, q=0.195, r=1.0),
}

# variance-only, both channels available in this repo's BreakSpec (r ==
# "variance" / observation-noise; q == "state_var" / state-innovation)
SCENARIOS = {
    "variance_x1.5": dict(kind="variance", time_frac=0.5, vol_mult=1.5),
    "variance_x3": dict(kind="variance", time_frac=0.5, vol_mult=3.0),
    "qvar_x1.5": dict(kind="state_var", time_frac=0.5, vol_mult=1.5),
    "qvar_x3": dict(kind="state_var", time_frac=0.5, vol_mult=3.0),
}


def _standardized_posttrain(Y: np.ndarray) -> np.ndarray:
    """Same standardization exp08 uses for PELT (and raw_cusum uses
    causally): post-training segment standardized by training-prefix
    mean/std, so ICSS gets no more information than the causal
    detectors' training split allows."""
    mu = Y[:N_TRAIN].mean()
    sd = max(Y[:N_TRAIN].std(ddof=1), 1e-12)
    return (Y[N_TRAIN:] - mu) / sd


def icss_fires(Y: np.ndarray, crit: float) -> bool:
    seg = _standardized_posttrain(Y)
    return len(icss_breakpoints(seg, crit=crit)) > 0


def icss_localizes(Y: np.ndarray, crit: float, break_time: int) -> bool:
    seg = _standardized_posttrain(Y)
    b_rel = break_time - N_TRAIN
    return any(abs(b - b_rel) <= WINDOW for b in icss_breakpoints(seg, crit=crit))


def icss_fires_rate(Ys: list[np.ndarray], crit: float) -> float:
    return float(np.mean([icss_fires(Y, crit) for Y in Ys]))


def calibrate_crit(null_dgp, n_reps: int, far_target: float, seed0: int,
                   lo: float = 0.0, hi: float = 1.0,
                   n_bisect: int = 14) -> tuple[float, float]:
    """Bisect crit so P(icss_fires | null) ~= far_target (higher crit
    -> fewer accepted breakpoints -> lower FAR, monotone -- same
    bisection direction as exp08's calibrate_pen)."""
    Ys = [null_dgp.sample(T, seed=seed0 + i).Y for i in range(n_reps)]
    if icss_fires_rate(Ys, lo) < far_target:
        raise RuntimeError(f"lo={lo} crit already below target FAR "
                           f"{far_target}; narrow the bisection bracket")
    if icss_fires_rate(Ys, hi) > far_target:
        raise RuntimeError(f"hi={hi} crit insufficient to reach FAR "
                           f"{far_target}; widen the bisection bracket")

    def far_at(crit: float) -> float:
        return float(np.mean([icss_fires(Y, crit) for Y in Ys]))

    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if far_at(mid) > far_target:
            lo = mid
        else:
            hi = mid
    crit = 0.5 * (lo + hi)
    return crit, far_at(crit)


def main(n_reps: int = 300) -> None:
    t0 = time.time()
    far_rows, rows = [], []
    for arena_name, arena_p in ARENAS.items():
        null_dgp = AR1StateDGP(**arena_p, breaks=[])
        crit, far_emp = calibrate_crit(null_dgp, n_reps, FAR_TARGET,
                                       SEEDS["calibration"])
        far_rows.append(dict(arena=arena_name, crit=crit,
                             far_target=FAR_TARGET, far_empirical=far_emp))
        print(f"[{time.time()-t0:6.0f}s] calibrated ICSS {arena_name}: "
              f"crit={crit:.4f} FAR={far_emp:.3f}", flush=True)

        for scen_name, spec_kwargs in SCENARIOS.items():
            spec = BreakSpec(**spec_kwargs)
            dgp = AR1StateDGP(**arena_p, breaks=[spec])
            b_time = spec.time(T)
            hits = [icss_localizes(dgp.sample(T, seed=SEEDS["evaluation"] + i).Y,
                                   crit, b_time)
                    for i in range(n_reps)]
            rate = float(np.mean(hits))
            se = float(np.std(hits, ddof=1) / np.sqrt(len(hits)))
            rows.append(dict(arena=arena_name, scenario=scen_name,
                             localize_rate=rate, localize_rate_se=se,
                             n=n_reps, window=WINDOW))
            print(f"[{time.time()-t0:6.0f}s] {arena_name} {scen_name}: "
                  f"localize={rate:.2f}", flush=True)

    pd.DataFrame(far_rows).to_csv(
        "paper_assets/exp25_icss_far_calibration.csv", index=False)
    pd.DataFrame(rows).to_csv(
        "paper_assets/exp25_icss_results.csv", index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote paper_assets/exp25_icss_*",
          flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
