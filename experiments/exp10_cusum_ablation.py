"""exp10 -- lsc_kalman_cusum construction ablation for the Table 2
flagship cell (SNR 0.5, phi=0.95, level 3 sigma), run to explain a
detection-rate discrepancy against an external one-sided, known-
parameter reimplementation (0.966) that does not match Table 2's
0.554.

Uses the SAME repo machinery as grid_v1 (lsc.diagnostics.alarms.calibrate,
lsc.diagnostics.features.break_pressure, lsc.eval.metrics,
lsc.theory.steady_state_innovations) on the exact grid_v1.yaml cell
(T=500, train_frac=0.25 -> n_train=125, far_target=0.05, n_reps=500,
arena ar1_snr0.5: phi=0.95, q=0.04875, r=1.0, scenario level_3s,
seeds calibration=100000 / evaluation=200000), varying only two axes
that the external reimplementation did not hold fixed:

  sidedness       -- Table 2's lsc_kalman_cusum is break_pressure(),
                     the two-sided max(g+, g-) Page CUSUM (k=0.5),
                     calibrated as one combined statistic (no FAR
                     split across arms). A one-sided variant here
                     uses only the g+ arm.
  parameterization -- Table 2 fits (phi, q, r) by MLE on the 125-obs
                     training prefix, per replication. The known-
                     parameter variant uses the true arena (phi, q, r)
                     directly via the steady-state Kalman filter
                     (lsc.theory.steady_state_innovations), same as
                     Propositions 1-2's assumption.

Four cells: (a) two-sided/estimated [= lsc_kalman_cusum as reported],
(b) one-sided/estimated, (c) two-sided/known, (d) one-sided/known.

Usage: python experiments/exp10_cusum_ablation.py
Output: paper_assets/exp10_cusum_ablation.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.diagnostics.alarms import calibrate
from lsc.diagnostics.features import break_pressure
from lsc.eval.detectors import _mask_train, make_innovation_cusum_detector
from lsc.eval.metrics import detection_outcome, summarize_detection
from lsc.models import KalmanModel
from lsc.theory import steady_state_innovations

T, N_TRAIN, FAR, N_REPS, K = 500, 125, 0.05, 500, 0.5
SEED_CAL, SEED_EVAL = 100_000, 200_000
PHI, Q, R = 0.95, 0.04875, 1.0  # arena ar1_snr0.5, configs/grid_v1.yaml


def one_sided_path(innovations: np.ndarray, k: float = K,
                   warmup: int = 10) -> np.ndarray:
    """break_pressure's own g+ recursion, positive arm only."""
    out = np.full(len(innovations), np.nan)
    gp = 0.0
    for t, e in enumerate(innovations):
        if not np.isfinite(e):
            e = 0.0
        gp = max(0.0, gp + e - k)
        if t >= warmup:
            out[t] = gp
    return out


def score_two_sided_estimated(Y: np.ndarray) -> np.ndarray:
    return make_innovation_cusum_detector(lambda: KalmanModel("ar1"),
                                          N_TRAIN)(Y)


def score_one_sided_estimated(Y: np.ndarray) -> np.ndarray:
    est = KalmanModel("ar1").fit_filter(Y, n_train=N_TRAIN)
    return _mask_train(one_sided_path(est.innovations), N_TRAIN)


def score_two_sided_known(Y: np.ndarray) -> np.ndarray:
    e = steady_state_innovations(Y, PHI, Q, R)
    return _mask_train(break_pressure(e, k=K), N_TRAIN)


def score_one_sided_known(Y: np.ndarray) -> np.ndarray:
    e = steady_state_innovations(Y, PHI, Q, R)
    return _mask_train(one_sided_path(e), N_TRAIN)


VARIANTS = {
    "a_two_sided_estimated": score_two_sided_estimated,
    "b_one_sided_estimated": score_one_sided_estimated,
    "c_two_sided_known": score_two_sided_known,
    "d_one_sided_known": score_one_sided_known,
}


def main() -> None:
    null_dgp = AR1StateDGP(phi=PHI, q=Q, r=R)
    break_dgp = AR1StateDGP(phi=PHI, q=Q, r=R, breaks=[
        BreakSpec(kind="level", time_frac=0.5, magnitude=3.0)])
    break_time = break_dgp.breaks[0].time(T)

    rows = []
    for label, fn in VARIANTS.items():
        det = calibrate(label, fn, null_dgp, T, n_reps=N_REPS, far=FAR,
                        seed0=SEED_CAL)
        outcomes = [detection_outcome(
            det.alarm_time(break_dgp.sample(T, seed=SEED_EVAL + i).Y),
            break_time, T) for i in range(N_REPS)]
        summ = summarize_detection(outcomes)
        rows.append(dict(variant=label, threshold=round(det.threshold, 4),
                         detect_rate=summ["detect_rate"],
                         detect_rate_se=round(summ["detect_rate_se"], 4)))
        print(f"{label:26s} threshold={det.threshold:.4f}  "
              f"detect_rate={summ['detect_rate']:.4f} "
              f"(se={summ['detect_rate_se']:.4f})")

    df = pd.DataFrame(rows)
    df.to_csv("paper_assets/exp10_cusum_ablation.csv", index=False)
    print("wrote paper_assets/exp10_cusum_ablation.csv")


if __name__ == "__main__":
    main()
