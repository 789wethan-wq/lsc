"""exp31 -- formal test of "the phi-sweep and SNR-sweep are one
experiment" (SPEC R3 M2, pre-registered in experiments/CHANGELOG.md
2026-07-25 BEFORE this script was run). Replaces the current two-
decimal eyeball match (Delta = 0.11 vs 0.11 at SNR 0.5; 0.07 vs 0.07 at
SNR ~ 2) with a formal comparison at both matched induced-SNR points.

Point 1 (SNR = 0.5): grid_v5's `ar1_snr0.5` arena (phi=0.95,
q=0.04875, r=1.0) and grid_v8's `ar1_phi0.95` arena (phi=0.95,
q=0.04875, r=1.0) are the SAME DGP parameterization evaluated with the
SAME calibration/evaluation seed bases -- not two independent
experiments that happen to agree, the identical computation appearing
in both grids by construction (grid_v8's phi=0.95 anchor was chosen to
reproduce the SNR=0.5 body arena). This script verifies that identity
directly against the committed CSVs rather than assuming it, and does
NOT run a hypothesis test there (there is nothing independent to test).

Point 2 (SNR = 2.0 vs. grid_v8's induced SNR = 2.45 at phi=0.99):
genuinely different (phi, q) parameterizations, so a shared-seed paired
test does not apply (the same seed integer drives different transition
dynamics under different phi/q -- not exchangeable pairs). Falls back
to the pre-registered permutation test: reconstruct per-replicate
detection outcomes for both cells by rerunning raw_var_cusum /
arima_var_cusum through each grid's ORIGINAL config and seed bases
(exp19's methodology -- verified against the published aggregate rates
before trusting the reconstruction), pool the 2*n_reps raw-minus-arima
per-replicate differences, permute the cell-A/cell-B group labels
n_perm times, and report the two-sided p-value for |Delta_A - Delta_B|
against that permutation null. Run separately per break size (x1.5,
x3), matching Table 4's own structure.

Usage: python experiments/exp31_phi_snr_sweep_equivalence_test.py [n_perm]
Output: paper_assets/exp31_point1_identity_check.csv,
        paper_assets/exp31_point2_permutation_test.csv
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

METHODS = ("raw_var_cusum", "arima_var_cusum")
SCENARIOS = ("qvar_x1.5", "qvar_x3")
PERM_SEED = 20260725


def point1_identity_check() -> pd.DataFrame:
    v5 = pd.read_csv(A / "grid_v5_qbreak_results.csv")
    v8 = pd.read_csv(A / "grid_v8_phiqbreak_results.csv")
    a = v5[(v5.arena == "ar1_snr0.5") & (v5.scenario.isin(SCENARIOS))
          & (v5.method.isin(METHODS))].sort_values(["scenario", "method"]).reset_index(drop=True)
    b = v8[(v8.arena == "ar1_phi0.95") & (v8.scenario.isin(SCENARIOS))
          & (v8.method.isin(METHODS))].sort_values(["scenario", "method"]).reset_index(drop=True)
    rows = []
    for scen in SCENARIOS:
        av = a[a.scenario == scen].set_index("method")
        bv = b[b.scenario == scen].set_index("method")
        delta_a = float(av.loc["raw_var_cusum", "detect_rate"] - av.loc["arima_var_cusum", "detect_rate"])
        delta_b = float(bv.loc["raw_var_cusum", "detect_rate"] - bv.loc["arima_var_cusum", "detect_rate"])
        identical = bool(np.array_equal(av["detect_rate"].values, bv["detect_rate"].values))
        rows.append(dict(scenario=scen, delta_snr_sweep=delta_a, delta_phi_sweep=delta_b,
                         bit_identical=identical))
    return pd.DataFrame(rows)


def _reconstruct_cell(config_path: Path, arena_name: str, n_reps: int
                      ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float]]:
    """Rerun raw_var_cusum/arima_var_cusum for one arena's two q-break
    scenarios through the config's own seed bases (exp19 methodology).
    Returns {scenario: {method: bool array of length n_reps}} and the
    reproduced aggregate detect_rate per (scenario, method), for the
    against-published check."""
    cfg = yaml.safe_load(config_path.read_text())
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

    outcomes: dict[str, dict[str, np.ndarray]] = {}
    rates: dict[str, float] = {}
    for scen in SCENARIOS:
        scen_cfg = cfg["scenarios"][scen]
        breaks = build_breaks(scen_cfg)
        dgp = build_dgp(arena_cfg, breaks)
        break_time = breaks[0].time(T)
        outcomes[scen] = {}
        for m in METHODS:
            arr = np.empty(n_reps, dtype=bool)
            for i in range(n_reps):
                Y = dgp.sample(T, seed=seeds["evaluation"] + i).Y
                out = detection_outcome(det[m].alarm_time(Y), break_time, T)
                arr[i] = out["detected"]
            outcomes[scen][m] = arr
            rates[f"{scen}|{m}"] = float(arr.mean())
    return outcomes, rates


def _check_against_published(rates: dict[str, float], published_path: Path,
                             arena_name: str) -> bool:
    pub = pd.read_csv(published_path)
    all_ok = True
    for key, rate in rates.items():
        scen, method = key.split("|")
        row = pub[(pub.arena == arena_name) & (pub.scenario == scen) & (pub.method == method)]
        if not len(row):
            all_ok = False
            continue
        pub_rate = float(row.detect_rate.iloc[0])
        if not np.isclose(rate, pub_rate, atol=1e-9):
            all_ok = False
    return all_ok


def permutation_test(d_a: np.ndarray, d_b: np.ndarray, n_perm: int, seed: int) -> dict:
    """Two-sided permutation p-value for mean(d_a) - mean(d_b), pooling
    and reshuffling group labels (unpaired, since cell A and cell B are
    different DGP parameterizations -- not exchangeable per-replicate
    pairs)."""
    rng = np.random.default_rng(seed)
    n_a, n_b = len(d_a), len(d_b)
    pooled = np.concatenate([d_a, d_b])
    observed = float(d_a.mean() - d_b.mean())
    perm_stats = np.empty(n_perm)
    for p in range(n_perm):
        rng.shuffle(pooled)
        perm_stats[p] = pooled[:n_a].mean() - pooled[n_a:].mean()
    p_value = float(np.mean(np.abs(perm_stats) >= abs(observed)))
    return dict(observed_diff=observed, p_value=p_value,
               se_a=float(d_a.std(ddof=1) / np.sqrt(n_a)),
               se_b=float(d_b.std(ddof=1) / np.sqrt(n_b)),
               n_a=n_a, n_b=n_b)


def point2_permutation_test(n_reps: int, n_perm: int) -> pd.DataFrame:
    t0 = time.time()
    outcomes_a, rates_a = _reconstruct_cell(CONFIGS / "grid_v5_qbreak.yaml", "ar1_snr2.0", n_reps)
    ok_a = _check_against_published(rates_a, A / "grid_v5_qbreak_results.csv", "ar1_snr2.0")
    print(f"[{time.time()-t0:6.0f}s] cell A (grid_v5 ar1_snr2.0) reproduced published rates: {ok_a}",
          flush=True)

    outcomes_b, rates_b = _reconstruct_cell(CONFIGS / "grid_v8_phiqbreak.yaml", "ar1_phi0.99", n_reps)
    ok_b = _check_against_published(rates_b, A / "grid_v8_phiqbreak_results.csv", "ar1_phi0.99")
    print(f"[{time.time()-t0:6.0f}s] cell B (grid_v8 ar1_phi0.99) reproduced published rates: {ok_b}",
          flush=True)

    if not (ok_a and ok_b):
        print("!!! reconstruction did not match published aggregates -- "
              "do NOT trust the permutation test below without investigating.")

    rows = []
    for scen in SCENARIOS:
        d_a = (outcomes_a[scen]["raw_var_cusum"].astype(float)
              - outcomes_a[scen]["arima_var_cusum"].astype(float))
        d_b = (outcomes_b[scen]["raw_var_cusum"].astype(float)
              - outcomes_b[scen]["arima_var_cusum"].astype(float))
        result = permutation_test(d_a, d_b, n_perm, seed=PERM_SEED)
        result.update(scenario=scen, n_reps=n_reps, n_perm=n_perm,
                      cellA_reproduced=ok_a, cellB_reproduced=ok_b)
        rows.append(result)
        print(f"[{time.time()-t0:6.0f}s] {scen}: Delta_A={d_a.mean():+.3f} "
              f"(SE {result['se_a']:.4f}) Delta_B={d_b.mean():+.3f} (SE {result['se_b']:.4f}) "
              f"observed diff={result['observed_diff']:+.3f} p={result['p_value']:.4f}", flush=True)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
    n_reps = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    print("=== Point 1 (SNR=0.5): identity check, no test applies ===")
    p1 = point1_identity_check()
    print(p1.to_string(index=False))
    p1.to_csv(A / "exp31_point1_identity_check.csv", index=False)

    print("\n=== Point 2 (SNR=2.0 vs induced SNR=2.45): permutation test ===")
    p2 = point2_permutation_test(n_reps, n_perm)
    p2.to_csv(A / "exp31_point2_permutation_test.csv", index=False)

    print(f"\nwrote {A}/exp31_point1_identity_check.csv, {A}/exp31_point2_permutation_test.csv")
