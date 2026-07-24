"""exp19_paired_se_grid_v8.py -- paired per-replicate standard errors for
Table 4's raw-vs-ARIMA advantage Delta on grid_v8_phiqbreak.

Table 4 (`PAPER_DRAFT.md`) reports Delta = detect(raw_var_cusum) -
detect(arima_var_cusum) per (phi, break size) cell and currently cites
a conservative, INDEPENDENCE-ASSUMING bound SE(Delta) <= 0.032 for
every cell. Raw and ARIMA are not independent draws: both are scored
on the SAME simulated Y path per replicate (`lsc.eval.runner.run`
draws `dgp.sample(T, seed=seeds["evaluation"] + i)` once per i and
scores every method against it), so the true SE of the paired
difference is very likely smaller than the conservative bound.

Per-replicate detection outcomes are NOT retained anywhere on disk:
`lsc.eval.runner.run` computes an `outcomes` list per (T, arena,
scenario, method) cell, immediately reduces it to
`summarize_detection(outcomes)`, and only the aggregated dict is
written to `grid_v8_phiqbreak_results.{csv,parquet}` -- the raw
per-replicate booleans are discarded. So this script reconstructs them
by RE-RUNNING raw_var_cusum and arima_var_cusum through the identical
code path (`lsc.eval.runner.build_dgp` / `build_detector`,
`lsc.diagnostics.alarms.calibrate`) with the SAME config
(`configs/grid_v8_phiqbreak.yaml`) and the SAME seed bases (calibration
100_000+i, evaluation 200_000+i) as the original run -- not a rerun
with a fresh seed, which would not reproduce the original pairing.
Both detectors are deterministic given Y (raw_var_cusum has no fitting
step at all; arima_var_cusum's `ARIMA(...).fit()` has no random
restarts or seeded search -- see `lsc/benchmarks/arima.py`), so this
reconstruction is bit-for-bit the same computation that produced the
published detect_rate numbers, not an approximation of it -- verified
below by comparing the reproduced aggregate rates against the
already-published `grid_v8_phiqbreak_results.csv`.

Usage: python experiments/exp19_paired_se_grid_v8.py
Output: prints, per cell, the reproduced rates (vs. published, as a
reproducibility check), the paired SE(Delta), an independence-assuming
bound using the ACTUAL observed p_raw/p_arima, and the paper's current
conservative worst-case (p=0.5) bound; writes
paper_assets/exp19_paired_se_grid_v8.csv.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lsc.diagnostics.alarms import calibrate
from lsc.eval.metrics import detection_outcome
from lsc.eval.runner import build_breaks, build_detector, build_dgp

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "grid_v8_phiqbreak.yaml"
PUBLISHED_PATH = REPO_ROOT / "paper_assets" / "grid_v8_phiqbreak_results.csv"
OUT_PATH = REPO_ROOT / "paper_assets" / "exp19_paired_se_grid_v8.csv"

METHODS = ("raw_var_cusum", "arima_var_cusum")


def worst_case_bound(n: int) -> float:
    """Conservative bound assuming independence AND p=0.5 (max variance
    for a Bernoulli proportion) for both detectors: SE(raw) <=
    sqrt(0.25/n), SE(arima) <= sqrt(0.25/n), SE(Delta) <=
    sqrt(SE(raw)^2 + SE(arima)^2) = sqrt(0.5/n). This is the bound
    Table 4 currently cites as <= 0.032 at n=500."""
    return float(np.sqrt(0.5 / n))


def independence_bound_actual(p_raw: float, p_arima: float, n: int) -> float:
    """Independence-assuming bound using the ACTUAL observed rates
    (tighter than the worst-case p=0.5 bound whenever p != 0.5)."""
    return float(np.sqrt(p_raw * (1 - p_raw) / n + p_arima * (1 - p_arima) / n))


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    n_reps = int(cfg["n_reps"])
    seeds = cfg["seeds"]
    T = cfg["T_values"][0]
    n_train = int(round(cfg["train_frac"] * T))
    published = pd.read_csv(PUBLISHED_PATH)

    t0 = time.time()
    rows = []
    for arena_name, arena_cfg in cfg["arenas"].items():
        null_dgp = build_dgp(arena_cfg, [])
        det = {}
        for method in METHODS:
            fn = build_detector(method, arena_cfg, null_dgp, T, n_train,
                                n_scale_reps=min(50, n_reps))
            det[method] = calibrate(method, fn, null_dgp, T, n_reps=n_reps,
                                    far=cfg["far_target"], seed0=seeds["calibration"])
        print(f"[{time.time()-t0:6.0f}s] {arena_name}: "
              f"thr raw={det['raw_var_cusum'].threshold:.3f} "
              f"arima={det['arima_var_cusum'].threshold:.3f}", flush=True)

        for scen_name, scen_cfg in cfg["scenarios"].items():
            breaks = build_breaks(scen_cfg)
            dgp = build_dgp(arena_cfg, breaks)
            break_time = breaks[0].time(T)

            detected = {m: np.empty(n_reps, dtype=bool) for m in METHODS}
            for i in range(n_reps):
                Y = dgp.sample(T, seed=seeds["evaluation"] + i).Y
                for m in METHODS:
                    outcome = detection_outcome(det[m].alarm_time(Y), break_time, T)
                    detected[m][i] = outcome["detected"]

            p_raw = float(detected["raw_var_cusum"].mean())
            p_arima = float(detected["arima_var_cusum"].mean())
            d = detected["raw_var_cusum"].astype(float) - detected["arima_var_cusum"].astype(float)
            paired_se = float(d.std(ddof=1) / np.sqrt(n_reps))
            delta = p_raw - p_arima

            pub = published[(published.arena == arena_name) & (published.scenario == scen_name)]
            pub_raw = pub[pub.method == "raw_var_cusum"].detect_rate
            pub_arima = pub[pub.method == "arima_var_cusum"].detect_rate
            pub_raw = float(pub_raw.iloc[0]) if len(pub_raw) else float("nan")
            pub_arima = float(pub_arima.iloc[0]) if len(pub_arima) else float("nan")
            reproduced_ok = np.isclose(p_raw, pub_raw, atol=1e-9) and np.isclose(p_arima, pub_arima, atol=1e-9)

            rows.append(dict(
                arena=arena_name, phi=float(arena_cfg["phi"]), scenario=scen_name,
                n_reps=n_reps, detect_raw=p_raw, detect_arima=p_arima,
                published_detect_raw=pub_raw, published_detect_arima=pub_arima,
                reproduced_exactly=reproduced_ok,
                delta=delta,
                se_delta_paired=paired_se,
                se_delta_independence_actual=independence_bound_actual(p_raw, p_arima, n_reps),
                se_delta_independence_worst_case=worst_case_bound(n_reps),
            ))
            print(f"  {scen_name}: Delta={delta:+.3f}  "
                  f"SE_paired={paired_se:.4f}  "
                  f"SE_indep(actual p)={independence_bound_actual(p_raw, p_arima, n_reps):.4f}  "
                  f"SE_indep(worst-case)={worst_case_bound(n_reps):.4f}  "
                  f"reproduced={reproduced_ok}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    all_reproduced = bool(out.reproduced_exactly.all())
    print(f"\nAll cells reproduced published detect_rate exactly: {all_reproduced}")
    if not all_reproduced:
        print("!!! reproduction mismatch -- do not trust the paired SEs below "
              "without investigating the discrepancy first.")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
