"""exp36 -- combined windowed statistic on a mixed-channel two-event
sequence (SPEC R4 M5, pre-registered in experiments/CHANGELOG.md
2026-07-25 BEFORE this script was run).

Sec 7 has two SEPARATE bounded-memory (MOSUM-style) fixes: windowed_raw_
cusum (mean-shift, exp04's level_up_down/level_then_var second-event
fix) and windowed_raw_var (variance-ratio, exp27's var_up_down
second-event fix) -- but never both running SIMULTANEOUSLY on a
mixed-channel sequence (level break then variance break, or the
reverse), the paper's own motivating real-world case (a level shift
followed by a volatility regime change, or vice versa).

New make_combined_windowed_detector (lsc/eval/detectors.py): one score
path, max(windowed_raw_cusum_score, windowed_raw_var_score) -- so a
single calibrated threshold covers both channels.

Two orderings, arena/protocol identical to exp04/exp27 (spec-SNR 0.5,
phi=0.95, T=500, n_train=125, breaks at t_frac 0.4/0.7, re-arm
rearm_frac=0.5/refractory=20, greedy one-to-one match window=100,
calibration=100000/evaluation=200000/far_check=300000):
  level_then_var: level +3sigma_ref at 0.4, obs-noise x3 at 0.7
                  (exp04's OWN level_then_var scenario, rerun here
                  with the new detector set -- exp04 never tested
                  windowed_raw_var or the combined detector on it)
  var_then_level: obs-noise x3 at 0.4, level +3sigma_ref at 0.7
                  (NEW ordering, not tested anywhere else)

Detectors: raw_var_cusum (fixed-baseline reference), windowed_raw_cusum
(mean-only), windowed_raw_var (variance-only), windowed_combined (new).

Usage: python experiments/exp36_combined_windowed_multibreak.py [n_reps]
Output: paper_assets/exp36_combined_windowed_multibreak.csv
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.diagnostics.alarms import calibrate, empirical_far
from lsc.eval.detectors import (
    make_combined_windowed_detector,
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

ARENA = dict(phi=0.95, q=0.04875, r=1.0)  # spec-SNR 0.5, same as exp04/exp27

SCENARIOS = {
    "level_then_var": [
        BreakSpec(kind="level", time_frac=0.4, magnitude=3.0),
        BreakSpec(kind="variance", time_frac=0.7, vol_mult=3.0),
    ],
    "var_then_level": [
        BreakSpec(kind="variance", time_frac=0.4, vol_mult=3.0),
        BreakSpec(kind="level", time_frac=0.7, magnitude=3.0),
    ],
}


def main(n_reps: int = 500) -> None:
    t0 = time.time()
    null_dgp = AR1StateDGP(**ARENA, breaks=[])
    factories = {
        "raw_var_cusum": make_raw_var_cusum_detector(N_TRAIN),
        "windowed_raw_cusum": make_windowed_raw_cusum_detector(N_TRAIN),
        "windowed_raw_var": make_windowed_raw_var_cusum_detector(N_TRAIN),
        "windowed_combined": make_combined_windowed_detector(N_TRAIN),
    }

    far_rows = []
    detectors = {}
    for name, fn in factories.items():
        det = calibrate(name, fn, null_dgp, T, n_reps=n_reps,
                        far=FAR_TARGET, seed0=SEEDS["calibration"])
        far = empirical_far(det, null_dgp, T, n_reps=n_reps, seed0=SEEDS["far_check"])
        detectors[name] = det
        far_rows.append(dict(method=name, threshold=det.threshold,
                             far_target=FAR_TARGET, far_empirical=far))
        print(f"[{time.time()-t0:6.0f}s] calibrated {name}: "
              f"thr={det.threshold:.3f} FAR={far:.3%}", flush=True)

    rows = []
    for scen_name, breaks in SCENARIOS.items():
        dgp = AR1StateDGP(**ARENA, breaks=breaks)
        b_times = [b.time(T) for b in breaks]
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
            rows.append(dict(scenario=scen_name, method=name, **summ))
            print(f"[{time.time()-t0:6.0f}s] {scen_name} {name}: "
                  f"recall_break1={summ['recall_break1']:.3f} "
                  f"recall_break2={summ['recall_break2']:.3f} "
                  f"precision={summ['precision']:.3f} F1={summ['f1']:.3f}", flush=True)

    pd.DataFrame(far_rows).to_csv(
        "paper_assets/exp36_combined_windowed_multibreak_far.csv", index=False)
    pd.DataFrame(rows).to_csv(
        "paper_assets/exp36_combined_windowed_multibreak.csv", index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote paper_assets/exp36_combined_windowed_multibreak*",
          flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
