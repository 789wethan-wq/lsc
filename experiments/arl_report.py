"""M5 — ARL0/ARL1 vocabulary alongside FAR (SPEC R1 §M5.1).

No new simulations: derived from the existing FAR-calibration tables and
detection parquets.

* ARL0 (in-control average run length) — the SPC/quickest-detection
  translation of the calibrated per-observation false-alarm rate. With
  the monitored window of length L = T - n_train and an empirical
  window-FAR α (P[>=1 false alarm in the window], from
  *_far_calibration.csv), the per-observation false-alarm hazard is
  p = 1 - (1-α)^(1/L) and ARL0 = 1/p. For small α this is ≈ L/α, the
  standard "one false alarm every L/α observations" reading. This is the
  quantity SPC control charts are designed around and that the
  ARL-matching convention (Basseville & Nikiforov 1993) equalizes across
  methods — here we equalize α instead, which is the same target
  expressed as a window-FAR.
* ARL1 (out-of-control ARL) — the post-break detection delay. We report
  mean_delay_detected (mean delay conditional on detection, already in
  the parquet); it is a lower-frame on the unconditional delay because
  missed paths (censored) are excluded, so we report the detection rate
  beside it.

Output: paper_assets/arl_table.csv (+ printed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

A = "paper_assets"
T, N_TRAIN = 500, 125
L = T - N_TRAIN  # monitored window length


def arl0_from_far(alpha: float) -> float:
    alpha = min(max(alpha, 1e-9), 0.999999)
    p = 1.0 - (1.0 - alpha) ** (1.0 / L)
    return 1.0 / p


def main() -> None:
    # ARL0 from every calibrated detector we have a FAR table for
    sources = ["grid_v1", "grid_v4_varbench", "grid_v5_qbreak",
               "grid_v6_phisweep", "grid_v7_llevel"]
    far_rows = []
    for src in sources:
        try:
            f = pd.read_csv(f"{A}/{src}_far_calibration.csv")
        except FileNotFoundError:
            continue
        f = f[f["T"] == T]
        for _, r in f.iterrows():
            far_rows.append(dict(source=src, arena=r.arena, method=r.method,
                                 far_empirical=r.far_empirical,
                                 arl0=arl0_from_far(r.far_empirical)))
    arl0 = pd.DataFrame(far_rows)

    # ARL1 (mean detection delay) on the canonical arena for the headline
    # detectors, from grid_v1
    arl1_rows = []
    try:
        g1 = pd.read_parquet(f"{A}/grid_v1_results.parquet")
        g1 = g1[(g1["T"] == T)
                & g1.method.isin(["raw_cusum", "lsc_kalman_cusum",
                                  "lsc_composite"])
                & g1.scenario.isin(["level_3s", "variance_x3"])]
        for _, r in g1.iterrows():
            arl1_rows.append(dict(arena=r.arena, scenario=r.scenario,
                                  method=r.method,
                                  detect_rate=round(r.detect_rate, 3),
                                  arl1_mean_delay=round(r.mean_delay_detected, 1)))
    except FileNotFoundError:
        pass
    arl1 = pd.DataFrame(arl1_rows)

    arl0.to_csv(f"{A}/arl_table.csv", index=False)
    arl1.to_csv(f"{A}/arl1_table.csv", index=False)

    print(f"monitored window L = {L} obs (T={T}, n_train={N_TRAIN})")
    print("\n=== ARL0 (in-control) at the 5% window-FAR target ===")
    print(f"  target FAR 0.05 -> ARL0 = {arl0_from_far(0.05):.0f} obs")
    print("  empirical, by detector (grid_v1 canonical arena SNR 0.5):")
    show = arl0[arl0.arena.astype(str).str.contains("snr0.5")]
    if len(show):
        print(show[["source", "method", "far_empirical", "arl0"]]
              .assign(arl0=lambda d: d.arl0.round(0)).to_string(index=False))
    print("\n=== ARL1 (mean detection delay, conditional on detection) ===")
    if len(arl1):
        print(arl1.to_string(index=False))
    print(f"\nwrote {A}/arl_table.csv, {A}/arl1_table.csv")


if __name__ == "__main__":
    main()
