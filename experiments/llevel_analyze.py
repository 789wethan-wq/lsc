"""M4 analysis — the local-level (RW-state) arena, demonstrated not
dismissed (SPEC R1 §M4).

Two facts replace the old one-clause dismissal, both read off grid_v7:
  (1) LEVEL detection is degenerate for EVERY method — a break in a
      random-walk state is absorbed by a well-specified filter as an
      ordinary large innovation, and the raw-Y CUSUM has no fixed
      baseline (its calibrated threshold is ~1e3, orders of magnitude
      above the AR(1) arena's, and it still calibrates hot).
  (2) VARIANCE detection is NOT degenerate, but whitening is now
      MANDATORY: the raw z^2 statistic sits at chance (threshold ~1e4,
      meaningless on a nonstationary series) while the ARIMA-differencing
      and Kalman rungs detect. The complement of the AR(1) result, where
      a raw variance CUSUM could win when observation noise dominates.

Output: paper_assets/grid_v7_llevel_summary.csv,
        paper_assets/grid_v7_llevel_degeneracy.png.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

A = "paper_assets"
M = ["raw_cusum", "lsc_kalman_cusum", "raw_var_cusum", "arima_var_cusum",
     "lsc_composite"]
LBL = {"raw_cusum": "raw\nCUSUM", "lsc_kalman_cusum": "Kalman\ninnov",
       "raw_var_cusum": "raw\nvar", "arima_var_cusum": "ARIMA\nvar",
       "lsc_composite": "composite"}


def main() -> None:
    df = pd.read_parquet(f"{A}/grid_v7_llevel_results.parquet")
    far = pd.read_csv(f"{A}/grid_v7_llevel_far_calibration.csv")
    det = df.pivot_table(index=["scenario", "arena"], columns="method",
                         values="detect_rate", aggfunc="first")[M]
    det.to_csv(f"{A}/grid_v7_llevel_summary.csv")

    thr = far.pivot_table(index="arena", columns="method",
                          values="threshold", aggfunc="first")[M]
    print("=== local-level arena: level breaks are degenerate for all "
          "methods (detect ~ FAR) ===")
    print(det.loc[["level_1s", "level_3s"]].round(3).to_string())
    print("\n=== but variance breaks ARE detectable — only by whitened "
          "rungs (raw z^2 at chance) ===")
    print(det.loc[["variance_x1.5", "variance_x3"]].round(3).to_string())
    print("\n=== raw detectors' thresholds are astronomical on "
          "nonstationary Y (vs AR(1) O(10-200)) ===")
    print(thr[["raw_cusum", "raw_var_cusum", "arima_var_cusum",
               "lsc_composite"]].round(0).to_string())

    _figure(df)
    print(f"\nwrote {A}/grid_v7_llevel_summary.csv, "
          f"{A}/grid_v7_llevel_degeneracy.png")


def _figure(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arena = "ll_snr0.5"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, scen, title in [
        (axes[0], "level_3s", "(a) 3σ level break: degenerate for all"),
        (axes[1], "variance_x1.5",
         "(b) ×1.5 variance break: only whitened rungs detect")]:
        sub = df[(df.scenario == scen) & (df.arena == arena)]
        vals = [float(sub[sub.method == m].detect_rate.iloc[0]) for m in M]
        ses = [float(sub[sub.method == m].detect_rate_se.iloc[0]) for m in M]
        colors = ["C7", "C7", "C7", "C0", "C0"]  # grey=raw/absorbed, blue=whiten
        ax.bar(range(len(M)), vals, yerr=ses, color=colors, capsize=3)
        ax.axhline(0.05, color="r", ls=":", lw=1, label="5% FAR")
        ax.set_xticks(range(len(M)))
        ax.set_xticklabels([LBL[m] for m in M], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("detect rate")
    fig.suptitle("Local-level (random-walk state) arena, SNR 0.5 — "
                 "whitening is mandatory when Y is nonstationary", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{A}/grid_v7_llevel_degeneracy.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
