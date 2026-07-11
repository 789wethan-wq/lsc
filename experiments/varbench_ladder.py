"""Assemble grid_v4 varbench outputs (SPEC addendum §3/§6).

1. Concatenate grid_v4_varbench_core + grid_v4_varbench_T results into
   the single tidy parquet `grid_v4_varbench_results.parquet` (+ CSV),
   same schema as every other grid; likewise the FAR-calibration table.
2. Build the whitening-ladder table (standalone CSV deliverable):
   detect rates (with MC SEs) for raw / ARIMA-whitened /
   Kalman-whitened rungs on the variance scenarios. Latent-rung rows
   are joined from the published parquets — grid_v1 (composite,
   Gaussian SNR arenas), grid_v2_T (composite, T sweep),
   grid_v2_misspec (composite, t₅), grid_v3c_tail (lsc_tail_cusum,
   SNR 0.5 Gaussian + t₅) — all produced with the identical seed
   blocks, so rows are draw-for-draw comparable.

Output: paper_assets/grid_v4_varbench_results.parquet/.csv,
        paper_assets/grid_v4_varbench_far_calibration.csv,
        paper_assets/ladder_table.csv (+ printed table).
"""
from __future__ import annotations

import pandas as pd

A = "paper_assets"


def main() -> None:
    core = pd.read_parquet(f"{A}/grid_v4_varbench_core_results.parquet")
    tsw = pd.read_parquet(f"{A}/grid_v4_varbench_T_results.parquet")
    df = pd.concat([core, tsw], ignore_index=True)
    df.to_parquet(f"{A}/grid_v4_varbench_results.parquet", index=False)
    df.to_csv(f"{A}/grid_v4_varbench_results.csv", index=False)
    far = pd.concat([pd.read_csv(f"{A}/grid_v4_varbench_core_far_calibration.csv"),
                     pd.read_csv(f"{A}/grid_v4_varbench_T_far_calibration.csv")],
                    ignore_index=True)
    far.to_csv(f"{A}/grid_v4_varbench_far_calibration.csv", index=False)

    # latent rung from the published grids (identical seed blocks)
    latent = []
    for src, methods in [("grid_v1", ["lsc_composite"]),
                         ("grid_v2_T", ["lsc_composite"]),
                         ("grid_v2_misspec", ["lsc_composite"]),
                         ("grid_v3c_tail", ["lsc_composite", "lsc_tail_cusum"])]:
        g = pd.read_parquet(f"{A}/{src}_results.parquet")
        g = g[g.method.isin(methods)].assign(source=src)
        latent.append(g)
    latent = pd.concat([df.assign(source="grid_v4_varbench")] + latent,
                       ignore_index=True)

    scen = latent[latent.scenario.str.startswith(("variance_",))]
    cols = ["T", "arena", "scenario", "method", "detect_rate",
            "detect_rate_se", "mean_delay_censored", "source"]
    ladder = (scen[cols]
              .drop_duplicates(["T", "arena", "scenario", "method"])
              .sort_values(["arena", "T", "scenario", "method"]))
    ladder.to_csv(f"{A}/ladder_table.csv", index=False)

    pivot = ladder[ladder["T"] == 500].pivot_table(
        index=["arena", "scenario"], columns="method",
        values="detect_rate", aggfunc="first")
    print(pivot.round(2).to_string())
    print(f"wrote {A}/grid_v4_varbench_results.parquet, {A}/ladder_table.csv")


if __name__ == "__main__":
    main()
