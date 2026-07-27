"""exp35 -- paired SE and z-statistic for Table 8's Kalman-vs-ARIMA
composite gaps (SPEC R4 M4, pre-registered in experiments/CHANGELOG.md
2026-07-25 BEFORE this script was run). Table 8's text currently argues
these gaps are "11-23 combined SEs" using a conservative independence-
assuming bound; this replaces that with the true paired SE, the same
upgrade exp19 gave Table 4's raw-vs-ARIMA Delta.

Both composite_kalman (published in grid_v1/grid_v5, method=
lsc_composite) and composite_arima (published in
exp20_composite_on_arima.csv) are scored on the SAME simulated Y path
per replicate: both use the standing calibration=100000/evaluation=
200000 seed bases with the SAME arena definition (phi, q, r), just a
different model fed to the identical composite machinery
(lsc.diagnostics.alarms.calibrate + lsc.eval.detectors.
make_composite_detector). Per-replicate outcomes are not persisted
anywhere (same situation exp19 documents for grid_v8): this script
reconstructs them by rerunning BOTH composites through their original
seed bases, and -- exactly like exp19 -- verifies the reconstructed
AGGREGATE detect_rate matches the published aggregate exactly before
trusting the reconstructed pairing (determinism argument: no random
restarts anywhere in KalmanModel.fit/ARIMAModel.fit, AR1StateDGP.sample
draws from a freshly-seeded independent RNG per call).

Three cells (user's choice, per spec: "at least the largest gap... and
one or two others"): r x1.5/SNR0.1 (the largest gap, 0.818 vs 0.226),
r x1.5/SNR2.0 (0.910 vs 0.632), q x3/SNR0.1 (0.438 vs 0.248, the
q-channel analog, smaller magnitude per the existing text).

Usage: python experiments/exp35_composite_paired_se.py [n_reps]
Output: paper_assets/exp35_composite_paired_se.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.diagnostics.alarms import calibrate
from lsc.diagnostics.features import COMPOSITE_V1
from lsc.eval.detectors import make_composite_detector
from lsc.eval.metrics import detection_outcome
from lsc.models import ARIMAModel, KalmanModel

REPO_ROOT = Path(__file__).resolve().parent.parent
A = REPO_ROOT / "paper_assets"
OUT_PATH = A / "exp35_composite_paired_se.csv"

PHI, R, T, N_TRAIN = 0.95, 1.0, 500, 125
FAR = 0.05
SEED_CAL, SEED_EVAL, SEED_SCALE = 100_000, 200_000, 900_000

# (channel, vol_mult, snr, published_kalman_path, arena_fmt, scenario_fmt)
CELLS = [
    ("r", 1.5, 0.1, A / "grid_v1_results.csv"),
    ("r", 1.5, 2.0, A / "grid_v1_results.csv"),
    ("q", 3.0, 0.1, A / "grid_v5_qbreak_results.csv"),
]


def _published_rate(path: Path, arena: str, scenario: str, method: str) -> float:
    df = pd.read_csv(path)
    m = df[(df.arena == arena) & (df.scenario == scenario) & (df.method == method)]
    return float(m.detect_rate.iloc[0]) if len(m) else float("nan")


def run_cell(channel: str, vol_mult: float, snr: float, kalman_path: Path,
            n_reps: int) -> dict:
    q = snr * (1 - PHI**2) * R
    null_dgp = AR1StateDGP(phi=PHI, q=q, r=R)
    kind = "variance" if channel == "r" else "state_var"
    break_dgp = AR1StateDGP(phi=PHI, q=q, r=R,
                            breaks=[BreakSpec(kind=kind, time_frac=0.5, vol_mult=vol_mult)])
    break_time = break_dgp.breaks[0].time(T)
    arena = f"ar1_snr{snr}"
    scen_vm = "3" if vol_mult == 3.0 else "1.5"
    scenario = ("variance_x{vm}" if channel == "r" else "qvar_x{vm}").format(vm=scen_vm)

    fn_kalman = make_composite_detector(lambda: KalmanModel("ar1"), null_dgp, T, N_TRAIN,
                                        n_scale_reps=min(50, n_reps),
                                        scale_seed0=SEED_SCALE, include=COMPOSITE_V1)
    det_kalman = calibrate("composite_kalman", fn_kalman, null_dgp, T, n_reps=n_reps,
                           far=FAR, seed0=SEED_CAL)

    fn_arima = make_composite_detector(lambda: ARIMAModel(), null_dgp, T, N_TRAIN,
                                       n_scale_reps=min(50, n_reps),
                                       scale_seed0=SEED_SCALE, include=COMPOSITE_V1)
    det_arima = calibrate("composite_arima", fn_arima, null_dgp, T, n_reps=n_reps,
                          far=FAR, seed0=SEED_CAL)

    detected_kalman = np.empty(n_reps, dtype=bool)
    detected_arima = np.empty(n_reps, dtype=bool)
    for i in range(n_reps):
        Y = break_dgp.sample(T, seed=SEED_EVAL + i).Y
        detected_kalman[i] = detection_outcome(det_kalman.alarm_time(Y), break_time, T)["detected"]
        detected_arima[i] = detection_outcome(det_arima.alarm_time(Y), break_time, T)["detected"]

    p_kalman = float(detected_kalman.mean())
    p_arima = float(detected_arima.mean())
    d = detected_kalman.astype(float) - detected_arima.astype(float)
    paired_se = float(d.std(ddof=1) / np.sqrt(n_reps))
    delta = p_kalman - p_arima
    z = delta / paired_se if paired_se > 0 else float("inf")

    pub_kalman = _published_rate(kalman_path, arena, scenario, "lsc_composite")
    # exp20's own CSV is keyed by (channel, snr, vol_mult), not arena/scenario
    exp20_df = pd.read_csv(A / "exp20_composite_on_arima.csv")
    m = exp20_df[(exp20_df.channel == channel) & np.isclose(exp20_df.snr, snr)
                & np.isclose(exp20_df.vol_mult, vol_mult)]
    pub_arima = float(m.detect_composite_arima.iloc[0]) if len(m) else float("nan")

    reproduced_kalman = bool(np.isclose(p_kalman, pub_kalman, atol=1e-9))
    reproduced_arima = bool(np.isclose(p_arima, pub_arima, atol=1e-9))

    independence_bound = float(np.sqrt(p_kalman * (1 - p_kalman) / n_reps
                                       + p_arima * (1 - p_arima) / n_reps))

    return dict(
        channel=channel, vol_mult=vol_mult, snr=snr, n_reps=n_reps,
        detect_kalman=p_kalman, detect_arima=p_arima, delta=delta,
        published_kalman=pub_kalman, published_arima=pub_arima,
        reproduced_kalman=reproduced_kalman, reproduced_arima=reproduced_arima,
        se_delta_paired=paired_se, z_paired=z,
        se_delta_independence=independence_bound,
        combined_se_independence=delta / independence_bound if independence_bound > 0 else float("inf"),
    )


if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    t0 = time.time()
    rows = []
    for channel, vol_mult, snr, kalman_path in CELLS:
        out = run_cell(channel, vol_mult, snr, kalman_path, n_reps)
        rows.append(out)
        print(f"[{time.time()-t0:6.0f}s] {channel} x{vol_mult} SNR{snr}: "
              f"kalman={out['detect_kalman']:.3f} (pub={out['published_kalman']:.3f}, "
              f"reproduced={out['reproduced_kalman']}) "
              f"arima={out['detect_arima']:.3f} (pub={out['published_arima']:.3f}, "
              f"reproduced={out['reproduced_arima']}) "
              f"Delta={out['delta']:+.3f} SE_paired={out['se_delta_paired']:.4f} "
              f"z={out['z_paired']:.2f} SE_indep={out['se_delta_independence']:.4f} "
              f"combined_SE_indep={out['combined_se_independence']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    all_ok = bool((df.reproduced_kalman & df.reproduced_arima).all())
    print(f"\nAll cells reproduced published aggregates exactly: {all_ok}")
    if not all_ok:
        print("!!! reproduction mismatch -- do not trust the paired SEs above.")
    print(f"wrote {OUT_PATH}")
