"""exp39 -- formal joint test of Table 4's "Delta peaks at phi=0.95,
recedes at phi=0.99" claim (SPEC R6 M2, pre-registered in
experiments/CHANGELOG.md 2026-07-26 BEFORE this script was run).

SCOPE NOTE on the request's "at the subtle x1.5 break, all three SNRs":
Table 4 (grid_v8_phiqbreak) has ONE q,r anchor (q=0.04875, r=1.0 FIXED)
with phi swept -- SNR is a DERIVED quantity of phi (induced SNR =
q/(r(1-phi^2))), not an independently-swept axis at each phi -- so
there is one point per phi, not three. This script adds phi in
{0.90, 0.97} to that SAME single-anchor sweep (now phi in {0.5, 0.8,
0.90, 0.95, 0.97, 0.99}), matching Table 4's actual convention exactly,
rather than also crossing 3 independent SNR arenas at every new phi
(a materially different, larger experiment the request's other text
does not otherwise describe).

phi in {0.95, 0.99} are reconstructed from the published grid_v8
results (per-replicate outcomes rerun through the ORIGINAL config/seed
bases, verified against the published aggregate before trusting the
reconstruction -- exp19/33's methodology). phi in {0.90, 0.97} are
genuinely new cells (q=0.04875, r=1.0, qvar_x1.5, standard seed blocks).

Joint test: phi=0.90/0.95/0.97/0.99 are DIFFERENT DGP parameterizations
-- same non-pairing argument as R3 M2/R4 M2 (shared seed integers do
not produce exchangeable draws across different phi). So this is NOT a
repeated-measures test across phi; each phi's n_reps=500 paired
per-replicate differences (raw_i - arima_i, itself a valid within-phi
pairing since raw/arima share a seed) are bootstrapped INDEPENDENTLY
(B=20000, seed 20260726): for each bootstrap draw, resample each phi's
500 differences with replacement, compute the bootstrap Delta(phi) =
mean of the resampled differences, and check whether Delta(0.95) >
Delta(0.90) AND Delta(0.95) > Delta(0.99) simultaneously in that draw.
The fraction of bootstrap draws satisfying BOTH inequalities jointly is
reported as the joint-peak probability; 1 minus that fraction is the
one-sided p-value against the null that the joint peak claim does not
hold.

Usage: python experiments/exp39_phi_peak_joint_test.py [n_reps] [n_boot]
Output: paper_assets/exp39_phi_peak_joint_test.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lsc.diagnostics.alarms import calibrate
from lsc.eval.metrics import detection_outcome
from lsc.eval.runner import build_breaks, build_detector, build_dgp
from lsc.dgp import AR1StateDGP, BreakSpec

REPO_ROOT = Path(__file__).resolve().parent.parent
A = REPO_ROOT / "paper_assets"
OUT_PATH = A / "exp39_phi_peak_joint_test.csv"

Q, R, T, N_TRAIN = 0.04875, 1.0, 500, 125
FAR = 0.05
SEED_CAL, SEED_EVAL = 100_000, 200_000
BOOT_SEED = 20260726

NEW_PHIS = (0.90, 0.97)
EXISTING_PHIS = (0.95, 0.99)  # reconstructed from grid_v8_phiqbreak
ALL_PHIS = (0.90, 0.95, 0.97, 0.99)


def per_replicate_diff_new(phi: float, n_reps: int) -> np.ndarray:
    """Fresh cell: raw_var_cusum - arima_var_cusum per replicate, q,r
    fixed at grid_v8's anchor, phi new."""
    null_dgp = AR1StateDGP(phi=phi, q=Q, r=R)
    from lsc.eval.detectors import make_arima_var_cusum_detector, make_raw_var_cusum_detector
    det_raw = calibrate("raw_var_cusum", make_raw_var_cusum_detector(N_TRAIN),
                        null_dgp, T, n_reps=n_reps, far=FAR, seed0=SEED_CAL)
    det_arima = calibrate("arima_var_cusum", make_arima_var_cusum_detector(N_TRAIN),
                          null_dgp, T, n_reps=n_reps, far=FAR, seed0=SEED_CAL)
    break_dgp = AR1StateDGP(phi=phi, q=Q, r=R,
                            breaks=[BreakSpec(kind="state_var", time_frac=0.5, vol_mult=1.5)])
    break_time = break_dgp.breaks[0].time(T)
    d = np.empty(n_reps)
    for i in range(n_reps):
        Y = break_dgp.sample(T, seed=SEED_EVAL + i).Y
        r_hit = detection_outcome(det_raw.alarm_time(Y), break_time, T)["detected"]
        a_hit = detection_outcome(det_arima.alarm_time(Y), break_time, T)["detected"]
        d[i] = float(r_hit) - float(a_hit)
    return d


def per_replicate_diff_existing(phi: float, n_reps: int) -> tuple[np.ndarray, bool]:
    """Reconstruct raw_var_cusum - arima_var_cusum per replicate for an
    ALREADY-PUBLISHED grid_v8_phiqbreak phi value, through the original
    config/seeds; verify against the published aggregate."""
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "grid_v8_phiqbreak.yaml").read_text())
    arena_name = f"ar1_phi{phi}"
    arena_cfg = cfg["arenas"][arena_name]
    seeds = cfg["seeds"]
    null_dgp = build_dgp(arena_cfg, [])
    det = {}
    for method in ("raw_var_cusum", "arima_var_cusum"):
        fn = build_detector(method, arena_cfg, null_dgp, T, N_TRAIN, n_scale_reps=min(50, n_reps))
        det[method] = calibrate(method, fn, null_dgp, T, n_reps=n_reps,
                                far=cfg["far_target"], seed0=seeds["calibration"])
    breaks = build_breaks(cfg["scenarios"]["qvar_x1.5"])
    dgp = build_dgp(arena_cfg, breaks)
    break_time = breaks[0].time(T)
    d = np.empty(n_reps)
    hits = {"raw_var_cusum": np.empty(n_reps, dtype=bool),
           "arima_var_cusum": np.empty(n_reps, dtype=bool)}
    for i in range(n_reps):
        Y = dgp.sample(T, seed=seeds["evaluation"] + i).Y
        for m in ("raw_var_cusum", "arima_var_cusum"):
            hits[m][i] = detection_outcome(det[m].alarm_time(Y), break_time, T)["detected"]
        d[i] = float(hits["raw_var_cusum"][i]) - float(hits["arima_var_cusum"][i])

    published = pd.read_csv(A / "grid_v8_phiqbreak_results.csv")
    ok = True
    for m in ("raw_var_cusum", "arima_var_cusum"):
        row = published[(published.arena == arena_name) & (published.scenario == "qvar_x1.5")
                       & (published.method == m)]
        pub_rate = float(row.detect_rate.iloc[0])
        ok = ok and np.isclose(hits[m].mean(), pub_rate, atol=1e-9)
    return d, bool(ok)


def bootstrap_joint_peak(diffs: dict[float, np.ndarray], n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = {phi: len(d) for phi, d in diffs.items()}
    boot_deltas = {phi: np.empty(n_boot) for phi in diffs}
    for phi, d in diffs.items():
        for b in range(n_boot):
            sample = rng.choice(d, size=n[phi], replace=True)
            boot_deltas[phi][b] = sample.mean()
    joint = (boot_deltas[0.95] > boot_deltas[0.90]) & (boot_deltas[0.95] > boot_deltas[0.99])
    joint_prob = float(joint.mean())
    return dict(
        observed_delta={phi: float(d.mean()) for phi, d in diffs.items()},
        se_delta={phi: float(d.std(ddof=1) / np.sqrt(len(d))) for phi, d in diffs.items()},
        joint_peak_probability=joint_prob,
        one_sided_p_value=1.0 - joint_prob,
        n_boot=n_boot,
    )


if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
    t0 = time.time()

    diffs = {}
    reproduced = {}
    for phi in EXISTING_PHIS:
        d, ok = per_replicate_diff_existing(phi, n_reps)
        diffs[phi] = d
        reproduced[phi] = ok
        print(f"[{time.time()-t0:6.0f}s] phi={phi} (reconstructed): "
              f"Delta={d.mean():+.3f} reproduced_published={ok}", flush=True)

    for phi in NEW_PHIS:
        d = per_replicate_diff_new(phi, n_reps)
        diffs[phi] = d
        reproduced[phi] = None  # no published cell to check against
        print(f"[{time.time()-t0:6.0f}s] phi={phi} (new): Delta={d.mean():+.3f}", flush=True)

    result = bootstrap_joint_peak(diffs, n_boot, BOOT_SEED)
    print(f"\n[{time.time()-t0:6.0f}s] observed Delta by phi: "
          f"{ {k: round(v,3) for k,v in result['observed_delta'].items()} }")
    print(f"joint peak probability (Delta(0.95) > both neighbors, bootstrap): "
          f"{result['joint_peak_probability']:.4f}")
    print(f"one-sided p-value: {result['one_sided_p_value']:.4f}")

    rows = [dict(phi=phi, n_reps=n_reps, delta=result["observed_delta"][phi],
                se_delta=result["se_delta"][phi], reproduced_published=reproduced[phi])
           for phi in ALL_PHIS]
    df = pd.DataFrame(rows)
    df["joint_peak_probability"] = result["joint_peak_probability"]
    df["one_sided_p_value"] = result["one_sided_p_value"]
    df["n_boot"] = n_boot
    df.to_csv(OUT_PATH, index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote {OUT_PATH}")
