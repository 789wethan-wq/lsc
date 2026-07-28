"""exp44 -- estimated-parameter Kalman variance CUSUM, the third
whitening-ladder rung (SPEC_R8_missing_experiments.md S2; pre-
registered in experiments/CHANGELOG.md 2026-07-27 BEFORE this script
was run).

The Sec 5 2x2 currently fills the "Kalman filter x single 3-arm CUSUM"
cell with arima_var_cusum's published numbers "by the ARMA(1,1)
equivalence" (exp07: exact at steady state, KNOWN parameters). This
measures whether that equivalence survives estimation: a new
est_kalman_var_cusum_score (lsc.benchmarks.variance) runs the
IDENTICAL three-arm statistic on an MLE-fit KalmanModel's causal
innovations instead of raw Y or ARIMA residuals.

Methodology (exp19/26/40's verify-before-trust pattern): reconstruct
raw_var_cusum and arima_var_cusum through the ORIGINAL grid_v4/grid_v5
config and seed bases, confirm the reconstruction matches the
published aggregate exactly, THEN add est_kalman_var_cusum on the
SAME evaluation draws (SPEC S0.2's explicit exception -- exp44 must
NOT get new evaluation seeds, or the paired comparison is lost).

Outputs:
  paper_assets/exp44_est_kalman_rung.csv       -- one row per cell
  paper_assets/exp44_innovation_tails.csv      -- per-cell tail summary
  paper_assets/exp44_perrep.csv                -- per-replicate long file (SPEC S0.1)

Usage: python experiments/exp44_est_kalman_rung.py [n_reps]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lsc.benchmarks.variance import est_kalman_var_cusum_score
from lsc.diagnostics.alarms import calibrate, empirical_far
from lsc.eval.detectors import _mask_train
from lsc.eval.metrics import detection_outcome
from lsc.eval.runner import build_breaks, build_detector, build_dgp

REPO_ROOT = Path(__file__).resolve().parent.parent
A = REPO_ROOT / "paper_assets"
CONFIGS = REPO_ROOT / "configs"

OUT_CELLS = A / "exp44_est_kalman_rung.csv"
OUT_TAILS = A / "exp44_innovation_tails.csv"
OUT_PERREP = A / "exp44_perrep.csv"

BASE_METHODS = ("raw_var_cusum", "arima_var_cusum")

CELLS = [
    dict(channel="r", config="grid_v4_varbench_core.yaml",
         results="grid_v4_varbench_core_results.csv",
         far_csv="grid_v4_varbench_core_far_calibration.csv",
         arena=f"ar1_snr{snr}", scenario=f"variance_x{vm}", snr=float(snr), vol_mult=float(vm))
    for snr in ("0.1", "0.5", "2.0") for vm in ("1.5", "3")
] + [
    dict(channel="q", config="grid_v5_qbreak.yaml",
         results="grid_v5_qbreak_results.csv",
         far_csv="grid_v5_qbreak_far_calibration.csv",
         arena=f"ar1_snr{snr}", scenario=f"qvar_x{vm}", snr=float(snr), vol_mult=float(vm))
    for snr in ("0.1", "0.5", "2.0") for vm in ("1.5", "3")
]


def _published_rate(results_path: Path, arena: str, scenario: str, method: str) -> float:
    df = pd.read_csv(results_path)
    m = df[(df.arena == arena) & (df.scenario == scenario) & (df.method == method)]
    return float(m.detect_rate.iloc[0]) if len(m) else float("nan")


def run_cell(cell: dict, n_reps: int) -> tuple[dict, dict, list[dict]]:
    cfg = yaml.safe_load((CONFIGS / cell["config"]).read_text())
    arena_cfg = cfg["arenas"][cell["arena"]]
    seeds = cfg["seeds"]
    T = cfg["T_values"][0]
    n_train = int(round(cfg["train_frac"] * T))
    far_target = cfg["far_target"]
    null_dgp = build_dgp(arena_cfg, [])
    spec = arena_cfg.get("kalman_spec", "llevel")

    det = {}
    for method in BASE_METHODS:
        fn = build_detector(method, arena_cfg, null_dgp, T, n_train, n_scale_reps=min(50, n_reps))
        det[method] = calibrate(method, fn, null_dgp, T, n_reps=n_reps,
                                far=far_target, seed0=seeds["calibration"])
    fn_est_kalman = lambda Y: est_kalman_var_cusum_score(Y, n_train=n_train, spec=spec)
    det["est_kalman_var_cusum"] = calibrate("est_kalman_var_cusum", fn_est_kalman, null_dgp, T,
                                            n_reps=n_reps, far=far_target, seed0=seeds["calibration"])
    methods = (*BASE_METHODS, "est_kalman_var_cusum")

    far_fresh = {m: empirical_far(det[m], null_dgp, T, n_reps=n_reps, seed0=seeds["far_check"])
                for m in methods}

    breaks = build_breaks(cfg["scenarios"][cell["scenario"]])
    dgp = build_dgp(arena_cfg, breaks)
    break_time = breaks[0].time(T)

    detected = {m: np.empty(n_reps, dtype=bool) for m in methods}
    alarm_idx = {m: np.full(n_reps, -1, dtype=int) for m in methods}
    score_max = {m: np.empty(n_reps) for m in methods}
    innov_gap_samples = []

    for i in range(n_reps):
        Y = dgp.sample(T, seed=seeds["evaluation"] + i).Y
        scores = {}
        for m in methods:
            s = det[m].score_fn(Y)
            scores[m] = s
            finite = s[np.isfinite(s)]
            score_max[m][i] = float(finite.max()) if len(finite) else float("-inf")
            at = det[m].alarm_time(Y)
            alarm_idx[m][i] = -1 if at is None else at
            detected[m][i] = detection_outcome(at, break_time, T)["detected"]
        # innovation-gap tail sample: est_kalman vs arima, on the SAME path,
        # both re-derived from their own frozen fits via the score functions'
        # underlying innovation series is not exposed by score_fn, so recompute
        # directly here (same n_train/spec as calibration above).
        from lsc.models import KalmanModel
        e_kalman = KalmanModel(spec).fit_filter(Y, n_train=n_train).innovations
        from lsc.benchmarks.arima import arima_standardized_residuals
        e_arima = arima_standardized_residuals(Y, n_train)
        gap = np.abs(e_kalman[n_train:] - e_arima[n_train:])
        innov_gap_samples.append(gap)

    p = {m: float(detected[m].mean()) for m in methods}
    reproduced = {m: bool(np.isclose(p[m], _published_rate(
        A / cell["results"], cell["arena"], cell["scenario"], m), atol=1e-9))
        for m in BASE_METHODS}

    d = detected["est_kalman_var_cusum"].astype(float) - detected["arima_var_cusum"].astype(float)
    se_paired = float(d.std(ddof=1) / np.sqrt(n_reps))
    outcome_corr = float(np.corrcoef(detected["est_kalman_var_cusum"].astype(float),
                                     detected["arima_var_cusum"].astype(float))[0, 1])

    all_gap = np.concatenate(innov_gap_samples)
    tail_row = dict(
        channel=cell["channel"], vol_mult=cell["vol_mult"], snr=cell["snr"],
        gap_p50=float(np.percentile(all_gap, 50)),
        gap_p95=float(np.percentile(all_gap, 95)),
        gap_p99=float(np.percentile(all_gap, 99)),
        maxscore_corr=float(np.corrcoef(score_max["est_kalman_var_cusum"],
                                        score_max["arima_var_cusum"])[0, 1]),
    )

    cell_row = dict(
        channel=cell["channel"], vol_mult=cell["vol_mult"], snr=cell["snr"], n_reps=n_reps,
        detect_est_kalman=p["est_kalman_var_cusum"], detect_arima=p["arima_var_cusum"],
        detect_raw=p["raw_var_cusum"],
        delta_kalman_arima=p["est_kalman_var_cusum"] - p["arima_var_cusum"],
        se_paired=se_paired, outcome_corr=outcome_corr,
        threshold_est_kalman=det["est_kalman_var_cusum"].threshold,
        threshold_arima=det["arima_var_cusum"].threshold,
        far_fresh_est_kalman=far_fresh["est_kalman_var_cusum"],
        far_fresh_arima=far_fresh["arima_var_cusum"],
        far_fresh_raw=far_fresh["raw_var_cusum"],
        reproduced_raw=reproduced["raw_var_cusum"],
        reproduced_arima=reproduced["arima_var_cusum"],
    )

    perrep_rows = []
    for i in range(n_reps):
        for m in methods:
            perrep_rows.append(dict(
                rep_id=i, arena=cell["arena"], scenario=cell["scenario"],
                channel=cell["channel"], vol_mult=cell["vol_mult"], snr=cell["snr"], phi=0.95,
                detector=m, detected=int(detected[m][i]),
                alarm_index=(None if alarm_idx[m][i] < 0 else int(alarm_idx[m][i])),
                score_max=score_max[m][i], threshold=det[m].threshold,
                seed=seeds["evaluation"] + i,
            ))

    return cell_row, tail_row, perrep_rows


def _load_existing(path: Path, key_cols: list[str]) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=key_cols)


def _already_done(existing: pd.DataFrame, channel: str, vol_mult: float, snr: float, n_reps: int | None = None):
    if existing.empty:
        return None
    m = existing[(existing.channel == channel) & np.isclose(existing.vol_mult, vol_mult)
                & np.isclose(existing.snr, snr)]
    if n_reps is not None:
        m = m[m.n_reps == n_reps]
    return m if len(m) else None


if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    t0 = time.time()

    existing_cells = _load_existing(OUT_CELLS, ["channel", "vol_mult", "snr", "n_reps"])
    existing_tails = _load_existing(OUT_TAILS, ["channel", "vol_mult", "snr"])
    existing_perrep = _load_existing(OUT_PERREP, ["channel", "vol_mult", "snr", "detector", "rep_id"])

    cell_rows, tail_rows, perrep_rows = [], [], []
    for cell in CELLS:
        cached = _already_done(existing_cells, cell["channel"], cell["vol_mult"], cell["snr"], n_reps)
        if cached is not None:
            cell_rows.append(cached.iloc[0].to_dict())
            tail_rows.append(_already_done(existing_tails, cell["channel"],
                                           cell["vol_mult"], cell["snr"]).iloc[0].to_dict())
            pr = existing_perrep[(existing_perrep.channel == cell["channel"])
                                 & np.isclose(existing_perrep.vol_mult, cell["vol_mult"])
                                 & np.isclose(existing_perrep.snr, cell["snr"])]
            perrep_rows.extend(pr.to_dict("records"))
            print(f"[{time.time()-t0:6.0f}s] {cell['channel']} x{cell['vol_mult']} "
                  f"SNR{cell['snr']}: reused", flush=True)
            continue
        cr, tr, pr = run_cell(cell, n_reps)
        cell_rows.append(cr)
        tail_rows.append(tr)
        perrep_rows.extend(pr)
        print(f"[{time.time()-t0:6.0f}s] {cell['channel']} x{cell['vol_mult']} SNR{cell['snr']}: "
              f"est_kalman={cr['detect_est_kalman']:.3f} arima={cr['detect_arima']:.3f} "
              f"raw={cr['detect_raw']:.3f} (Delta={cr['delta_kalman_arima']:+.3f}, "
              f"SE_paired={cr['se_paired']:.4f}) "
              f"reproduced=({cr['reproduced_raw']},{cr['reproduced_arima']})", flush=True)

    df_cells = pd.DataFrame(cell_rows).sort_values(["channel", "vol_mult", "snr"]).reset_index(drop=True)
    df_cells.to_csv(OUT_CELLS, index=False)
    df_tails = pd.DataFrame(tail_rows).sort_values(["channel", "vol_mult", "snr"]).reset_index(drop=True)
    df_tails.to_csv(OUT_TAILS, index=False)
    df_perrep = pd.DataFrame(perrep_rows)
    df_perrep.to_csv(OUT_PERREP, index=False)

    n_not_reproduced = int((~(df_cells.reproduced_raw & df_cells.reproduced_arima)).sum())
    n_h44_pass = int((df_cells.delta_kalman_arima.abs() <= 0.03).sum())
    print(f"\n[{time.time()-t0:6.0f}s] {len(df_cells)} cells; "
          f"{n_not_reproduced} did not reproduce published raw/arima aggregates exactly; "
          f"H44 (|Delta|<=0.03): {n_h44_pass}/{len(df_cells)} cells pass "
          f"(need >=10/12 for Outcome A)")
    print(f"wrote {OUT_CELLS}\nwrote {OUT_TAILS}\nwrote {OUT_PERREP}")
