"""exp13b -- export the real per-segment score PATHS (not just alarm
months) that real_data.py's rolling monitor produces internally but
never saves, so a future exp13 rework can build a genuinely
correlation-aware joint test instead of resampling alarm-month lists.

This is NOT a new statistical test and does not change any existing
result: it re-runs the exact same segmented loop, detector
construction, and calibration seeds as experiments/real_data.py::run
(seed0=100_000+1000*seg_id, n_cal=200, same SERIES config), and simply
also records det.score_fn(Y) and det.threshold per segment/method,
instead of only alarm_time. Reusing real_data.py's own SERIES dict,
fitted_null, and detector factories directly (not reimplementing them)
so this cannot silently diverge from what actually produced
paper_assets/rd_*_alarms.csv / rd_*_summary.csv (Table 6).

Per-segment scores are NOT on a common scale (each segment recalibrates
its own threshold from a fresh parametric-bootstrap null), so this also
exports each segment's threshold and score/threshold, which is the
comparable, stitchable quantity across segments/methods for a future
joint test.

Usage: python experiments/exp13b_export_scorepaths.py SERIES
Output: paper_assets/exp13b_{series}_scorepaths.csv
        (columns: segment, method, date, score, threshold, score_over_threshold)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "paper_assets"

from lsc.diagnostics.alarms import calibrate
from lsc.eval.detectors import (
    make_innovation_cusum_detector,
    make_raw_cusum_detector,
    make_raw_var_cusum_detector,
    make_tail_cusum_detector,
    make_composite_detector,
)
from lsc.models import KalmanModel
from real_data import SERIES, fitted_null, load_series


def run(series: str) -> None:
    t0 = time.time()
    cfg = dict(SERIES[series])
    g = load_series(cfg, live=False)
    NT, NM = cfg["n_train"], cfg["n_monitor"]
    T_seg = NT + NM
    n_cal = 200  # matches real_data.py's default, and rd_{series}_meta.csv's n_cal

    rows = []
    seg_id = 0
    for start in range(0, len(g) - T_seg + 1, NM):
        seg = g.iloc[start:start + T_seg]
        Y = seg.values
        null = fitted_null(Y[:NT])
        comp_fn = make_composite_detector(
            lambda: KalmanModel("ar1"), null, T_seg, NT, n_scale_reps=50)
        detectors = {
            "lsc_composite": comp_fn,
            "lsc_tail_cusum": make_tail_cusum_detector(
                lambda: KalmanModel("ar1"), NT),
            "lsc_kalman_cusum": make_innovation_cusum_detector(
                lambda: KalmanModel("ar1"), NT),
            "raw_cusum": make_raw_cusum_detector(NT),
            "raw_var_cusum": make_raw_var_cusum_detector(NT),
        }
        for mname, fn in detectors.items():
            det = calibrate(mname, fn, null, T_seg, n_reps=n_cal,
                            far=cfg.get("far", 0.05),
                            seed0=100_000 + 1000 * seg_id)
            score = det.score_fn(Y)
            monitored = score[NT:]
            for j, s in enumerate(monitored):
                rows.append(dict(
                    segment=seg_id, method=mname,
                    date=seg.index[NT + j], score=float(s) if np.isfinite(s) else np.nan,
                    threshold=det.threshold,
                    score_over_threshold=float(s) / det.threshold
                                          if np.isfinite(s) else np.nan))
        print(f"[{time.time()-t0:5.0f}s] segment {seg_id} "
              f"({seg.index[NT]:%Y-%m} monitored) done", flush=True)
        seg_id += 1

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / f"exp13b_{series}_scorepaths.csv"
    out.to_csv(out_path, index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote {out_path}: "
          f"{seg_id} segments x {len(detectors)} methods, "
          f"{len(out)} rows")


if __name__ == "__main__":
    run(sys.argv[1])
