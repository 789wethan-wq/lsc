"""exp16_aic_order_frequencies.py -- quantifies the near-unit-root AIC
order-selection behavior currently described only qualitatively in §5
and Appendix B ("rarely selects the exact (1,0,1)... prefers (1,0,0)
or a differencing (0,1,1)"), per the review's Missing Experiment 1 /
Minor Weakness 3: a table of selection frequencies by (phi, SNR), not
just a prose claim.

Reuses `lsc.benchmarks.arima.fit_arima_prefix` directly -- the exact
AIC-selected, training-prefix-frozen order-search already used to
produce every ARIMA-rung number in the paper (Sec 5, Appendix B) -- so
this tallies what that existing call actually picks, rather than
introducing a separate order-selection procedure that could disagree
with it. Grid matches configs/grid_v6_phisweep.yaml exactly: phi in
{0.5, 0.8, 0.95, 0.99}, SNR in {0.1, 0.5, 2.0} (q = SNR*(1-phi^2),
r=1.0), T=500, train_frac=0.25 (n_train=125), n_reps=500, seed0=100000
-- the same calibration-null seed convention used throughout this
project (exp11-exp15), not a new one.

Usage: python experiments/exp16_aic_order_frequencies.py [n_reps]
Output: paper_assets/exp16_aic_order_frequencies.csv (+ printed table)
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from lsc.benchmarks.arima import fit_arima_prefix
from lsc.dgp import AR1StateDGP

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "paper_assets" / "exp16_aic_order_frequencies.csv"

PHI_GRID = [0.5, 0.8, 0.95, 0.99]
SNR_GRID = [0.1, 0.5, 2.0]
T, TRAIN_FRAC, R = 500, 0.25, 1.0
N_TRAIN = int(round(TRAIN_FRAC * T))
SEED0 = 100_000  # matches grid_v6_phisweep.yaml's calibration seed


def order_selection_frequencies(phi: float, snr: float, n_reps: int) -> Counter:
    q = snr * (1 - phi**2) * R
    dgp = AR1StateDGP(phi=phi, q=q, r=R)
    counts: Counter = Counter()
    for i in range(n_reps):
        Y = dgp.sample(T, seed=SEED0 + i).Y
        order, _ = fit_arima_prefix(Y, N_TRAIN)
        counts[order] += 1
    return counts


def main(n_reps: int = 500) -> None:
    t0 = time.time()
    rows = []
    for phi in PHI_GRID:
        for snr in SNR_GRID:
            counts = order_selection_frequencies(phi, snr, n_reps)
            total = sum(counts.values())
            for order, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                rows.append(dict(phi=phi, snr=snr, order=str(order),
                                 count=n, frequency=n / total))
            summary = ", ".join(f"{k}={v/total:.1%}" for k, v in
                                sorted(counts.items(), key=lambda kv: -kv[1]))
            print(f"[{time.time()-t0:6.0f}s] phi={phi} SNR={snr}: {summary}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote {OUT_PATH}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
