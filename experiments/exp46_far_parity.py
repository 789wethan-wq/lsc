"""exp46 -- symmetric false-alarm-rate recalibration across every
Table 1-3 detector (SPEC_R8_missing_experiments.md S4; pre-registered
in experiments/CHANGELOG.md 2026-07-27 BEFORE this script was run).

Part A (must run/print first): reconciles the apparent sign
disagreement between Table 1's raw_cusum empirical FAR
(4.0/6.2/8.2% at SNR 0.1/0.5/2.0, from grid_v1_far_calibration.csv,
computed via lsc.eval.runner.run's own empirical_far call on the
far_check=300000 block at n_reps=500) and exp38's baseline
out-of-sample FAR (5.3/6.4/7.2%, same far_check=300000 block but
n_reps=2000). BOTH numbers are already out-of-sample (disjoint
far_check seeds from the calibration=100000 block in both cases) --
this checks directly, before trusting anything else in this script,
whether the disagreement is a genuine in-sample/out-of-sample
confound or simply finite-n sampling noise in the FAR ESTIMATE at
n=500 vs n=2000 (lsc-methodology-lessons #3: order-statistic
threshold exceedance has ~+-1.3pp sd at n=300 on heavy-tailed null
maxima). Recomputes raw_cusum's threshold (calibration seed0=100000,
n_reps=500, reproduces the published threshold exactly) and its
empirical FAR against the FIRST 500 vs the first 2000 draws of the
SAME far_check=300000 sequence.

Part B: for every detector in Tables 1-3 (raw_cusum, lsc_kalman_cusum,
lsc_state_cusum, lsc_composite -- Table 2 / level scenarios;
raw_var_cusum, arima_var_cusum, est_kalman_var_cusum, lsc_tail_cusum --
Table 3 / r-channel variance scenarios; lsc_composite scored on both),
at every grid_v1/grid_v4 arena (identical phi=0.95 / q / r params,
shared across configs): calibrate at 5000 reps (block 100000-104999,
a strict extension of the published 500), verify FAR on 2000 FRESH
nulls (block 330000-331999, disjoint from the standing 300000 block
so exp24's GARCH check is unaffected), take the FAR-matched threshold
as the fresh block's own 95th-percentile max score (the closed-form
limit of bisecting against that same fixed fresh sample -- iterating
a bisection search on a FIXED finite sample converges exactly to its
own order statistic, so this is not an approximation, just the
non-iterative form), and re-score Table 2/3's scenarios at all three
thresholds (q500 / q5000 / far-matched) using ONE detector-scoring
pass per evaluation replicate (thresholds compared post hoc against
the stored score path, not re-simulated per threshold).

Outputs:
  paper_assets/exp46_far_reconciliation.csv  -- Part A
  paper_assets/exp46_far_parity.csv          -- Part B, one row per (detector,arena)
  paper_assets/exp46_detect_matched.csv      -- Part B, one row per (detector,arena,scenario)
  paper_assets/exp46_perrep.csv              -- per-replicate long file (SPEC S0.1)

Usage: python experiments/exp46_far_parity.py [n_reps_large]
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.diagnostics.alarms import calibrate
from lsc.eval.detectors import make_raw_cusum_detector
from lsc.eval.metrics import detection_outcome
from lsc.eval.runner import build_detector

REPO_ROOT = Path(__file__).resolve().parent.parent
A = REPO_ROOT / "paper_assets"

OUT_RECONCILE = A / "exp46_far_reconciliation.csv"
OUT_PARITY = A / "exp46_far_parity.csv"
OUT_DETECT = A / "exp46_detect_matched.csv"
OUT_PERREP = A / "exp46_perrep.csv"

PHI, R, T, N_TRAIN = 0.95, 1.0, 500, 125
SEED_CAL, SEED_EVAL, SEED_FAR_STANDING, SEED_FAR_FRESH = 100_000, 200_000, 300_000, 330_000
N_REPS_LARGE_DEFAULT = 5000
N_REPS_Q500 = 500
N_REPS_FRESH = 2000
FAR_LO, FAR_HI = 0.045, 0.055

ARENAS = {
    "ar1_snr0.1": dict(snr=0.1, q=0.00975, r=R),
    "ar1_snr0.5": dict(snr=0.5, q=0.04875, r=R),
    "ar1_snr2.0": dict(snr=2.0, q=0.195, r=R),
}

LEVEL_SCENARIOS = {
    "level_0.5s": dict(kind="level", time_frac=0.5, magnitude=0.5),
    "level_1s": dict(kind="level", time_frac=0.5, magnitude=1.0),
    "level_3s": dict(kind="level", time_frac=0.5, magnitude=3.0),
}
VARIANCE_SCENARIOS = {
    "variance_x1.5": dict(kind="variance", time_frac=0.5, vol_mult=1.5),
    "variance_x3": dict(kind="variance", time_frac=0.5, vol_mult=3.0),
}

DETECTOR_SCENARIOS = {
    "raw_cusum": LEVEL_SCENARIOS,
    "lsc_kalman_cusum": LEVEL_SCENARIOS,
    "lsc_state_cusum": LEVEL_SCENARIOS,
    "lsc_composite": {**LEVEL_SCENARIOS, **VARIANCE_SCENARIOS},
    "raw_var_cusum": VARIANCE_SCENARIOS,
    "arima_var_cusum": VARIANCE_SCENARIOS,
    "est_kalman_var_cusum": VARIANCE_SCENARIOS,
    "lsc_tail_cusum": VARIANCE_SCENARIOS,
}


def part_a() -> pd.DataFrame:
    rows = []
    for arena, p in ARENAS.items():
        null_dgp = AR1StateDGP(phi=PHI, q=p["q"], r=p["r"])
        fn = make_raw_cusum_detector(N_TRAIN)
        det = calibrate("raw_cusum", fn, null_dgp, T, n_reps=N_REPS_Q500,
                        far=0.05, seed0=SEED_CAL)
        far_scores = np.array([
            fn(null_dgp.sample(T, seed=SEED_FAR_STANDING + i).Y)
            for i in range(N_REPS_FRESH)
        ])
        far_max = np.array([s[np.isfinite(s)].max() if np.isfinite(s).any() else -np.inf
                            for s in far_scores])
        far_n500 = float((far_max[:500] >= det.threshold).mean())
        far_n2000 = float((far_max[:2000] >= det.threshold).mean())
        rows.append(dict(arena=arena, snr=p["snr"], threshold=det.threshold,
                         far_n500=far_n500, far_n2000=far_n2000,
                         far_n500_se=float(np.sqrt(0.05 * 0.95 / 500)),
                         far_n2000_se=float(np.sqrt(0.05 * 0.95 / 2000))))
        print(f"Part A {arena}: threshold={det.threshold:.3f} "
              f"far@n=500={far_n500:.3%} (SE~{np.sqrt(0.05*0.95/500):.3%}) "
              f"far@n=2000={far_n2000:.3%} (SE~{np.sqrt(0.05*0.95/2000):.3%})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_RECONCILE, index=False)
    print(f"wrote {OUT_RECONCILE}\n")
    return df


def _score_max(score_fn, Y) -> float:
    s = score_fn(Y)
    finite = s[np.isfinite(s)]
    return float(finite.max()) if len(finite) else float("-inf")


def run_detector_arena(method: str, arena: str, p: dict, n_reps_large: int,
                       n_reps_fresh: int = N_REPS_FRESH
                       ) -> tuple[dict, list[dict], list[dict]]:
    arena_cfg = dict(dgp="AR1StateDGP", phi=PHI, q=p["q"], r=p["r"], kalman_spec="ar1")
    null_dgp = AR1StateDGP(phi=PHI, q=p["q"], r=p["r"])
    score_fn = build_detector(method, arena_cfg, null_dgp, T, N_TRAIN,
                              n_scale_reps=min(50, n_reps_large))

    calib_scores = np.array([
        _score_max(score_fn, null_dgp.sample(T, seed=SEED_CAL + i).Y)
        for i in range(n_reps_large)
    ])
    thresh_q500 = float(np.quantile(calib_scores[:N_REPS_Q500], 0.95))
    thresh_q5000 = float(np.quantile(calib_scores[:n_reps_large], 0.95))

    fresh_scores = np.array([
        _score_max(score_fn, null_dgp.sample(T, seed=SEED_FAR_FRESH + i).Y)
        for i in range(n_reps_fresh)
    ])
    thresh_matched = float(np.quantile(fresh_scores, 0.95))
    far_fresh_q500 = float((fresh_scores >= thresh_q500).mean())
    far_fresh_q5000 = float((fresh_scores >= thresh_q5000).mean())
    far_fresh_matched = float((fresh_scores >= thresh_matched).mean())

    parity_row = dict(
        detector=method, arena=arena, snr=p["snr"],
        n_reps_large_used=n_reps_large, n_reps_fresh_used=n_reps_fresh,
        thresh_q500=thresh_q500, thresh_q5000=thresh_q5000, thresh_far_matched=thresh_matched,
        far_fresh_q500=far_fresh_q500, far_fresh_q5000=far_fresh_q5000,
        far_fresh_matched=far_fresh_matched,
        matched_in_band=bool(FAR_LO <= far_fresh_matched <= FAR_HI),
    )

    detect_rows = []
    perrep_rows = []
    scenarios = DETECTOR_SCENARIOS[method]
    for scen_name, scen_kwargs in scenarios.items():
        break_dgp = AR1StateDGP(phi=PHI, q=p["q"], r=p["r"],
                                breaks=[BreakSpec(**scen_kwargs)])
        break_time = break_dgp.breaks[0].time(T)
        det_q500 = det_q5000 = det_matched = 0
        for i in range(N_REPS_Q500):
            Y = break_dgp.sample(T, seed=SEED_EVAL + i).Y
            s = score_fn(Y)
            finite_mask = np.isfinite(s)
            hits = np.nonzero(finite_mask & (s >= thresh_q500))[0]
            at_q500 = int(hits[0]) if len(hits) else None
            hits5 = np.nonzero(finite_mask & (s >= thresh_q5000))[0]
            at_q5000 = int(hits5[0]) if len(hits5) else None
            hitsm = np.nonzero(finite_mask & (s >= thresh_matched))[0]
            at_matched = int(hitsm[0]) if len(hitsm) else None

            d500 = detection_outcome(at_q500, break_time, T)["detected"]
            d5000 = detection_outcome(at_q5000, break_time, T)["detected"]
            dmatched = detection_outcome(at_matched, break_time, T)["detected"]
            det_q500 += int(d500)
            det_q5000 += int(d5000)
            det_matched += int(dmatched)

            score_max_i = float(s[finite_mask].max()) if finite_mask.any() else float("-inf")
            perrep_rows.append(dict(
                rep_id=i, arena=arena, scenario=scen_name, channel="r", vol_mult=np.nan,
                snr=p["snr"], phi=PHI, detector=method, detected=int(dmatched),
                alarm_index=at_matched, score_max=score_max_i, threshold=thresh_matched,
                seed=SEED_EVAL + i,
            ))
        n = N_REPS_Q500
        detect_rows.append(dict(
            detector=method, arena=arena, snr=p["snr"], scenario=scen_name,
            detect_q500=det_q500 / n, detect_q5000=det_q5000 / n, detect_matched=det_matched / n,
        ))

    return parity_row, detect_rows, perrep_rows


DEFAULT_BUDGET = dict(n_reps_large=N_REPS_LARGE_DEFAULT, n_reps_fresh=N_REPS_FRESH)
# arima_var_cusum: ~2s/replicate (AIC search over 5 orders) makes the full
# 5000/2000 spec budget ~13h for this ONE detector across 3 arenas (measured
# via a 100-rep timing probe, 2026-07-27) -- reduced per the author's explicit
# choice (AskUserQuestion, "Full spec, arima_var_cusum reduced"), logged in
# CHANGELOG.md before this ran. Every other detector runs at the full spec
# budget (fit-free or Kalman-MLE-only, both orders of magnitude cheaper).
DETECTOR_BUDGET = {
    "arima_var_cusum": dict(n_reps_large=1000, n_reps_fresh=1000),
}


def _budget_for(method: str) -> dict:
    return DETECTOR_BUDGET.get(method, DEFAULT_BUDGET)


def _load_existing_parity() -> pd.DataFrame:
    if OUT_PARITY.exists():
        return pd.read_csv(OUT_PARITY)
    return pd.DataFrame(columns=["detector", "arena", "n_reps_large_used", "n_reps_fresh_used"])


def part_b() -> None:
    t0 = time.time()
    existing = _load_existing_parity()
    existing_detect = pd.read_csv(OUT_DETECT) if OUT_DETECT.exists() else pd.DataFrame()
    existing_perrep = pd.read_csv(OUT_PERREP) if OUT_PERREP.exists() else pd.DataFrame()

    parity_rows, detect_rows, perrep_rows = [], [], []
    plan = [(m, a) for m in DETECTOR_SCENARIOS for a in ARENAS]
    for method, arena in plan:
        budget = _budget_for(method)
        cached = existing[(existing.detector == method) & (existing.arena == arena)
                          & (existing.n_reps_large_used == budget["n_reps_large"])
                          & (existing.n_reps_fresh_used == budget["n_reps_fresh"])]
        if len(cached):
            pr = cached.iloc[0].to_dict()
            parity_rows.append(pr)
            if len(existing_detect):
                detect_rows.extend(existing_detect[(existing_detect.detector == method)
                                                    & (existing_detect.arena == arena)].to_dict("records"))
            if len(existing_perrep):
                perrep_rows.extend(existing_perrep[(existing_perrep.detector == method)
                                                    & (existing_perrep.arena == arena)].to_dict("records"))
            print(f"[{time.time()-t0:6.0f}s] {method} {arena}: reused "
                  f"(n_large={budget['n_reps_large']}, n_fresh={budget['n_reps_fresh']})", flush=True)
            continue

        pr, dr, rr = run_detector_arena(method, arena, ARENAS[arena],
                                        budget["n_reps_large"], budget["n_reps_fresh"])
        parity_rows.append(pr)
        detect_rows.extend(dr)
        perrep_rows.extend(rr)
        print(f"[{time.time()-t0:6.0f}s] {method} {arena} (n_large={budget['n_reps_large']}, "
              f"n_fresh={budget['n_reps_fresh']}): "
              f"thr500={pr['thresh_q500']:.3f} thr5000={pr['thresh_q5000']:.3f} "
              f"thr_matched={pr['thresh_far_matched']:.3f} | "
              f"far500={pr['far_fresh_q500']:.3%} far5000={pr['far_fresh_q5000']:.3%} "
              f"far_matched={pr['far_fresh_matched']:.3%} "
              f"(in_band={pr['matched_in_band']})", flush=True)

        # persist incrementally -- this run is many hours; do not lose
        # progress on interruption.
        pd.DataFrame(parity_rows).to_csv(OUT_PARITY, index=False)
        pd.DataFrame(detect_rows).to_csv(OUT_DETECT, index=False)
        pd.DataFrame(perrep_rows).to_csv(OUT_PERREP, index=False)

    df_parity = pd.DataFrame(parity_rows)
    df_parity.to_csv(OUT_PARITY, index=False)
    pd.DataFrame(detect_rows).to_csv(OUT_DETECT, index=False)
    pd.DataFrame(perrep_rows).to_csv(OUT_PERREP, index=False)

    n_in_band = int(df_parity.matched_in_band.sum())
    print(f"\n[{time.time()-t0:6.0f}s] {len(df_parity)} (detector,arena) rows; "
          f"{n_in_band}/{len(df_parity)} far_fresh_matched in [4.5%,5.5%]")
    print(f"wrote {OUT_PARITY}\nwrote {OUT_DETECT}\nwrote {OUT_PERREP}")


if __name__ == "__main__":
    part_a()
    part_b()
