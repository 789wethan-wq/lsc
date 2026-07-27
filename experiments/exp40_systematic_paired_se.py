"""exp40 -- systematic paired SEs across every raw-vs-ARIMA whitening-
ladder cell in Tables 3, 3b, 3c (the full r-channel phi-sweep), and 4
(SPEC R6 M3, pre-registered in experiments/CHANGELOG.md 2026-07-26
BEFORE this script was run).

Reuses exp19/30/33's exact methodology (rerun through the ORIGINAL
config/seed bases, verify the reconstructed aggregate matches the
already-published rate exactly before trusting the pairing) across
EVERY cell in:
  Table 3   grid_v4_varbench_core.yaml (r-channel) +
            grid_v5_qbreak.yaml (q-channel), phi=0.95,
            SNR in {0.1,0.5,2.0} x vol_mult in {1.5,3}      = 12 cells
  Table 3b/3c  grid_v9_r_phi99.yaml (phi=0.99) +
            grid_v9b_r_phi_lo.yaml (phi in {0.5,0.8}),
            r-channel, same SNR x vol_mult cross            = 18 cells
            (phi=0.95 already counted under Table 3, not
            re-reconstructed a second time)
  Table 4   grid_v8_phiqbreak.yaml, q-channel, phi in
            {0.1,0.5,0.7,0.85,0.95,0.99} x vol_mult in {1.5,3} = 12 cells

Total 42 cells. Both compared rungs (raw_var_cusum, arima_var_cusum)
share calibration/evaluation seeds within every cell (the paper's
draw-for-draw convention), so a true per-replicate pairing applies
throughout -- this is not assumed, it is the same determinism argument
exp19 verifies directly (no random restarts in either detector's
fitting step).

Falsifiable per the request: this reports whichever cells do NOT
tighten under pairing (paired SE >= independence-bound SE) exactly as
found, not smoothed to "pairing always helps."

Usage: python experiments/exp40_systematic_paired_se.py [n_reps]
Output: paper_assets/exp40_systematic_paired_se.csv (one row per cell:
        table, config, arena, scenario, delta, se_paired,
        se_independence, reproduced_raw, reproduced_arima)
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

REPO_ROOT = Path(__file__).resolve().parent.parent
A = REPO_ROOT / "paper_assets"
CONFIGS = REPO_ROOT / "configs"
OUT_PATH = A / "exp40_systematic_paired_se.csv"

METHODS = ("raw_var_cusum", "arima_var_cusum")


def _cells_from_config(config_name: str, results_name: str, scenarios: list[str]
                       ) -> list[tuple]:
    cfg = yaml.safe_load((CONFIGS / config_name).read_text())
    cells = []
    for arena_name in cfg["arenas"]:
        for scen_name in scenarios:
            if scen_name in cfg["scenarios"]:
                cells.append((config_name, results_name, arena_name, scen_name))
    return cells


def enumerate_cells() -> list[dict]:
    plan = []
    for cfg_name, res_name, scens, table in [
        ("grid_v4_varbench_core.yaml", "grid_v4_varbench_core_results.csv",
         ["variance_x1.5", "variance_x3"], "Table 3 (r)"),
        ("grid_v5_qbreak.yaml", "grid_v5_qbreak_results.csv",
         ["qvar_x1.5", "qvar_x3"], "Table 3 (q)"),
        ("grid_v9_r_phi99.yaml", "grid_v9_r_phi99_results.csv",
         ["variance_x1.5", "variance_x3"], "Table 3b (phi=0.99)"),
        ("grid_v9b_r_phi_lo.yaml", "grid_v9b_r_phi_lo_results.csv",
         ["variance_x1.5", "variance_x3"], "Table 3c (phi=0.5,0.8)"),
        ("grid_v8_phiqbreak.yaml", "grid_v8_phiqbreak_results.csv",
         ["qvar_x1.5", "qvar_x3"], "Table 4 (phi-sweep)"),
    ]:
        for cfg_name2, res2, arena, scen in _cells_from_config(cfg_name, res_name, scens):
            plan.append(dict(table=table, config=cfg_name2, results=res2,
                             arena=arena, scenario=scen))
    return plan


def _published_rate(results_path: Path, arena: str, scenario: str, method: str) -> float:
    df = pd.read_csv(results_path)
    m = df[(df.arena == arena) & (df.scenario == scenario) & (df.method == method)]
    return float(m.detect_rate.iloc[0]) if len(m) else float("nan")


def reconstruct_cell(config_name: str, results_name: str, arena_name: str,
                     scenario_name: str, n_reps: int) -> dict:
    cfg = yaml.safe_load((CONFIGS / config_name).read_text())
    arena_cfg = cfg["arenas"][arena_name]
    seeds = cfg["seeds"]
    T = cfg["T_values"][0]
    n_train = int(round(cfg["train_frac"] * T))
    null_dgp = build_dgp(arena_cfg, [])

    det = {}
    for method in METHODS:
        fn = build_detector(method, arena_cfg, null_dgp, T, n_train, n_scale_reps=min(50, n_reps))
        det[method] = calibrate(method, fn, null_dgp, T, n_reps=n_reps,
                                far=cfg["far_target"], seed0=seeds["calibration"])

    breaks = build_breaks(cfg["scenarios"][scenario_name])
    dgp = build_dgp(arena_cfg, breaks)
    break_time = breaks[0].time(T)

    detected = {m: np.empty(n_reps, dtype=bool) for m in METHODS}
    for i in range(n_reps):
        Y = dgp.sample(T, seed=seeds["evaluation"] + i).Y
        for m in METHODS:
            detected[m][i] = detection_outcome(det[m].alarm_time(Y), break_time, T)["detected"]

    p_raw = float(detected["raw_var_cusum"].mean())
    p_arima = float(detected["arima_var_cusum"].mean())
    d = detected["raw_var_cusum"].astype(float) - detected["arima_var_cusum"].astype(float)
    se_paired = float(d.std(ddof=1) / np.sqrt(n_reps))
    se_independence = float(np.sqrt(p_raw * (1 - p_raw) / n_reps + p_arima * (1 - p_arima) / n_reps))

    results_path = A / results_name
    pub_raw = _published_rate(results_path, arena_name, scenario_name, "raw_var_cusum")
    pub_arima = _published_rate(results_path, arena_name, scenario_name, "arima_var_cusum")

    return dict(
        config=config_name, arena=arena_name, scenario=scenario_name, n_reps=n_reps,
        detect_raw=p_raw, detect_arima=p_arima, delta=p_raw - p_arima,
        published_raw=pub_raw, published_arima=pub_arima,
        reproduced_raw=bool(np.isclose(p_raw, pub_raw, atol=1e-9)),
        reproduced_arima=bool(np.isclose(p_arima, pub_arima, atol=1e-9)),
        se_paired=se_paired, se_independence=se_independence,
        tightens_under_pairing=bool(se_paired < se_independence),
    )


def _load_existing() -> pd.DataFrame:
    if OUT_PATH.exists():
        return pd.read_csv(OUT_PATH)
    return pd.DataFrame(columns=["config", "arena", "scenario", "n_reps"])


def _already_done(existing, config, arena, scenario, n_reps):
    if existing.empty:
        return None
    m = existing[(existing.config == config) & (existing.arena == arena)
                & (existing.scenario == scenario) & (existing.n_reps == n_reps)]
    return m.iloc[0].to_dict() if len(m) else None


if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    t0 = time.time()
    plan = enumerate_cells()
    print(f"[{time.time()-t0:6.0f}s] {len(plan)} cells to process", flush=True)
    existing = _load_existing()
    rows = []
    for cell in plan:
        cached = _already_done(existing, cell["config"], cell["arena"], cell["scenario"], n_reps)
        if cached is not None:
            cached["table"] = cell["table"]
            rows.append(cached)
            print(f"[{time.time()-t0:6.0f}s] {cell['table']} {cell['arena']}/{cell['scenario']}: "
                  f"reused", flush=True)
            continue
        out = reconstruct_cell(cell["config"], cell["results"], cell["arena"],
                               cell["scenario"], n_reps)
        out["table"] = cell["table"]
        rows.append(out)
        print(f"[{time.time()-t0:6.0f}s] {cell['table']} {cell['arena']}/{cell['scenario']}: "
              f"Delta={out['delta']:+.3f} SE_paired={out['se_paired']:.4f} "
              f"SE_indep={out['se_independence']:.4f} "
              f"tightens={out['tightens_under_pairing']} "
              f"reproduced=({out['reproduced_raw']},{out['reproduced_arima']})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    n_not_tighten = int((~df.tightens_under_pairing).sum())
    n_not_reproduced = int((~(df.reproduced_raw & df.reproduced_arima)).sum())
    print(f"\n[{time.time()-t0:6.0f}s] {len(df)} cells total; "
          f"{n_not_tighten} did NOT tighten under pairing; "
          f"{n_not_reproduced} did not reproduce published aggregates exactly")
    print(f"wrote {OUT_PATH}")
