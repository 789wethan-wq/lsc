"""exp04 — multi-break paths, event-level F1 (hypotheses pre-registered
in CHANGELOG, 2026-07-11 entry).

Two breaks per path in the AR(1) arena (spec-SNR 0.5, T=500,
n_train=125): breaks at t=200 and t=350. Detectors are calibrated to 5%
FAR per path on the matched null exactly as in the single-break
experiments (first-alarm logic is unchanged), then evaluated with the
re-arm protocol (CalibratedDetector.alarm_times: re-arm once the score
drains below half the threshold and a 20-obs refractory has passed).
Alarms are matched to breaks one-to-one, greedily in time order, within
a 100-obs window; unmatched alarms are false positives (metrics.
multi_break_outcome). The null-path alarm-count distribution under the
re-arm protocol is reported alongside (protocol FAR context).

Scenarios:
  level_up_down  — level +3 sigma_ref at 0.4, level -3 at 0.7
  level_then_var — level +3 at 0.4, obs-noise x3 at 0.7
  var_up_down    — obs-noise x3 at 0.4, then x(1/3) at 0.7 (back to
                   baseline: the second event is a pure quieting)

Outputs: paper_assets/exp04_results.csv/.parquet,
exp04_far_calibration.csv, exp04_null_alarm_counts.csv.

Usage: python experiments/exp04_multibreak.py [n_reps]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.diagnostics.alarms import calibrate, empirical_far
from lsc.eval.detectors import (
    make_composite_detector,
    make_innovation_cusum_detector,
    make_raw_cusum_detector,
    make_state_cusum_detector,
)
from lsc.eval.metrics import multi_break_outcome, summarize_multi_break
from lsc.models import KalmanModel

T = 500
N_TRAIN = 125
FAR_TARGET = 0.05
WINDOW = 100
REARM_FRAC = 0.5
REFRACTORY = 20

SEEDS = dict(calibration=100_000, evaluation=200_000, far_check=300_000,
             feature_scales=900_000)

ARENA = dict(phi=0.95, q=0.04875, r=1.0)  # spec-SNR 0.5

SCENARIOS = {
    "level_up_down": [
        BreakSpec(kind="level", time_frac=0.4, magnitude=3.0),
        BreakSpec(kind="level", time_frac=0.7, magnitude=-3.0),
    ],
    "level_then_var": [
        BreakSpec(kind="level", time_frac=0.4, magnitude=3.0),
        BreakSpec(kind="variance", time_frac=0.7, vol_mult=3.0),
    ],
    "var_up_down": [
        BreakSpec(kind="variance", time_frac=0.4, vol_mult=3.0),
        BreakSpec(kind="variance", time_frac=0.7, vol_mult=1.0 / 3.0),
    ],
}


def main(n_reps: int = 500) -> None:
    t0 = time.time()
    null_dgp = AR1StateDGP(**ARENA, breaks=[])
    factories = {
        "lsc_composite": make_composite_detector(
            lambda: KalmanModel("ar1"), null_dgp, T, N_TRAIN,
            n_scale_reps=50, scale_seed0=SEEDS["feature_scales"]),
        "lsc_kalman_cusum": make_innovation_cusum_detector(
            lambda: KalmanModel("ar1"), N_TRAIN),
        "lsc_state_cusum": make_state_cusum_detector(
            lambda: KalmanModel("ar1"), N_TRAIN),
        "raw_cusum": make_raw_cusum_detector(N_TRAIN),
    }

    far_rows, null_rows, rows = [], [], []
    detectors = {}
    for name, fn in factories.items():
        det = calibrate(name, fn, null_dgp, T, n_reps=n_reps,
                        far=FAR_TARGET, seed0=SEEDS["calibration"])
        far = empirical_far(det, null_dgp, T, n_reps=n_reps,
                            seed0=SEEDS["far_check"])
        detectors[name] = det
        far_rows.append(dict(method=name, threshold=det.threshold,
                             far_target=FAR_TARGET, far_empirical=far))
        # alarm-count distribution on nulls under the re-arm protocol
        counts = [len(det.alarm_times(
                      null_dgp.sample(T, seed=SEEDS["far_check"] + i).Y,
                      rearm_frac=REARM_FRAC, refractory=REFRACTORY))
                  for i in range(n_reps)]
        null_rows.append(dict(method=name,
                              mean_null_alarms=float(np.mean(counts)),
                              max_null_alarms=int(np.max(counts)),
                              frac_ge1=float(np.mean(np.array(counts) >= 1)),
                              frac_ge2=float(np.mean(np.array(counts) >= 2))))
        print(f"[{time.time()-t0:6.0f}s] calibrated {name}: "
              f"thr={det.threshold:.3f} FAR={far:.3%} "
              f"null-alarms/path={np.mean(counts):.3f}", flush=True)

    for scen_name, breaks in SCENARIOS.items():
        dgp = AR1StateDGP(**ARENA, breaks=breaks)
        b_times = [b.time(T) for b in breaks]
        for name, det in detectors.items():
            outcomes = [
                multi_break_outcome(
                    det.alarm_times(
                        dgp.sample(T, seed=SEEDS["evaluation"] + i).Y,
                        rearm_frac=REARM_FRAC, refractory=REFRACTORY),
                    b_times, T, window=WINDOW)
                for i in range(n_reps)
            ]
            summ = summarize_multi_break(outcomes)
            # per-break recall: which of the two events gets seen
            summ["recall_break1"] = float(
                np.mean([o["matched"][0] for o in outcomes]))
            summ["recall_break2"] = float(
                np.mean([o["matched"][1] for o in outcomes]))
            rows.append(dict(scenario=scen_name, method=name, **summ))
            print(f"[{time.time()-t0:6.0f}s] {scen_name} {name}: "
                  f"F1={summ['f1']:.2f} recall={summ['recall']:.2f} "
                  f"precision={summ['precision']:.2f}", flush=True)

    pd.DataFrame(far_rows).to_csv(
        "paper_assets/exp04_far_calibration.csv", index=False)
    pd.DataFrame(null_rows).to_csv(
        "paper_assets/exp04_null_alarm_counts.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_parquet("paper_assets/exp04_results.parquet", index=False)
    df.to_csv("paper_assets/exp04_results.csv", index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote paper_assets/exp04_*", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
