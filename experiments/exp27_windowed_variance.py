"""exp27 -- windowed (bounded-memory) variance statistic for the
var_up_down second-event miss (peer review round 3, Missing
Experiments): exp04's existing bounded-memory MOSUM fix
(windowed_raw_cusum) was designed for MEAN shifts and, as already
documented ("Windowed-CUSUM fix, level->var / var->var 2nd event: no
improvement"), leaves the variance-channel second event undetected --
a pure variance change does not move a two-window MEAN-comparison
statistic. This adds windowed_raw_var_score
(lsc.benchmarks.variance), a two-window log-variance-ratio statistic,
and tests it on the EXACT SAME var_up_down scenario exp04 already
uses: obs-noise x3 at t=200, x(1/3) (back to baseline) at t=350 --
150-observation spacing, T=500, arena spec-SNR 0.5, identical seeds
and re-arm protocol (rearm_frac=0.5, refractory=20, match window=100).

Detectors: raw_var_cusum (single fixed-baseline reference, the
existing variance-ladder bottom rung), windowed_raw_cusum (the
existing MEAN-based bounded-memory fix, for direct comparison against
its documented 0.00 second-event recall), windowed_raw_var (the new
VARIANCE-based bounded-memory statistic under test).

Usage: python experiments/exp27_windowed_variance.py [n_reps]
Output: paper_assets/exp27_windowed_variance.csv
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.diagnostics.alarms import calibrate, empirical_far
from lsc.eval.detectors import (
    make_raw_var_cusum_detector,
    make_windowed_raw_cusum_detector,
    make_windowed_raw_var_cusum_detector,
)
from lsc.eval.metrics import multi_break_outcome, summarize_multi_break

T = 500
N_TRAIN = 125
FAR_TARGET = 0.05
WINDOW = 100
REARM_FRAC = 0.5
REFRACTORY = 20

SEEDS = dict(calibration=100_000, evaluation=200_000, far_check=300_000)

ARENA = dict(phi=0.95, q=0.04875, r=1.0)  # spec-SNR 0.5, same as exp04

BREAKS = [
    BreakSpec(kind="variance", time_frac=0.4, vol_mult=3.0),
    BreakSpec(kind="variance", time_frac=0.7, vol_mult=1.0 / 3.0),
]


def main(n_reps: int = 500) -> None:
    t0 = time.time()
    null_dgp = AR1StateDGP(**ARENA, breaks=[])
    factories = {
        "raw_var_cusum": make_raw_var_cusum_detector(N_TRAIN),
        "windowed_raw_cusum": make_windowed_raw_cusum_detector(N_TRAIN),
        "windowed_raw_var": make_windowed_raw_var_cusum_detector(N_TRAIN),
    }

    far_rows, rows = [], []
    detectors = {}
    for name, fn in factories.items():
        det = calibrate(name, fn, null_dgp, T, n_reps=n_reps,
                        far=FAR_TARGET, seed0=SEEDS["calibration"])
        far = empirical_far(det, null_dgp, T, n_reps=n_reps,
                            seed0=SEEDS["far_check"])
        detectors[name] = det
        far_rows.append(dict(method=name, threshold=det.threshold,
                             far_target=FAR_TARGET, far_empirical=far))
        print(f"[{time.time()-t0:6.0f}s] calibrated {name}: "
              f"thr={det.threshold:.3f} FAR={far:.3%}", flush=True)

    dgp = AR1StateDGP(**ARENA, breaks=BREAKS)
    b_times = [b.time(T) for b in BREAKS]
    for name, det in detectors.items():
        outcomes = [
            multi_break_outcome(
                det.alarm_times(dgp.sample(T, seed=SEEDS["evaluation"] + i).Y,
                               rearm_frac=REARM_FRAC, refractory=REFRACTORY),
                b_times, T, window=WINDOW)
            for i in range(n_reps)
        ]
        summ = summarize_multi_break(outcomes)
        summ["recall_break1"] = float(np.mean([o["matched"][0] for o in outcomes]))
        summ["recall_break2"] = float(np.mean([o["matched"][1] for o in outcomes]))
        rows.append(dict(scenario="var_up_down", method=name, **summ))
        print(f"[{time.time()-t0:6.0f}s] var_up_down {name}: "
              f"F1={summ['f1']:.3f} recall_break1={summ['recall_break1']:.3f} "
              f"recall_break2={summ['recall_break2']:.3f} "
              f"precision={summ['precision']:.3f}", flush=True)

    pd.DataFrame(far_rows).to_csv(
        "paper_assets/exp27_windowed_variance_far.csv", index=False)
    pd.DataFrame(rows).to_csv(
        "paper_assets/exp27_windowed_variance.csv", index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote paper_assets/exp27_windowed_variance*",
          flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
