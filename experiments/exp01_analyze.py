"""Summarize exp01 results: comparison table + detect-rate/delay plot.

Usage: python experiments/exp01_analyze.py [results.csv]
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

METHOD_ORDER = ["lsc_state_cusum", "lsc_composite", "lsc_kalman_cusum",
                "raw_cusum", "arima_cusum", "plain_hmm"]


def main(path: str = "paper_assets/exp01_results.csv") -> None:
    df = pd.read_csv(path)
    far = pd.read_csv(path.replace("results", "far_calibration"))
    print("=== Empirical FAR (target 5%) ===")
    print(far.to_string(index=False))

    for arena, g in df.groupby("arena"):
        print(f"\n=== arena: {arena} ===")
        piv = g.pivot_table(index="scenario", columns="method",
                            values=["detect_rate", "median_delay_detected",
                                    "mean_delay_censored"])
        for metric in ("detect_rate", "median_delay_detected", "mean_delay_censored"):
            cols = [m for m in METHOD_ORDER if m in piv[metric].columns]
            print(f"\n-- {metric} --")
            print(piv[metric][cols].round(2).to_string())

    # frontier plot: detect rate vs mean censored delay per scenario (ar1)
    g = df[df.arena == "ar1"]
    scens = g.scenario.unique()
    fig, axes = plt.subplots(1, len(scens), figsize=(4 * len(scens), 4),
                             sharey=True)
    for ax, scen in zip(axes, scens):
        sub = g[g.scenario == scen]
        for _, row in sub.iterrows():
            ax.errorbar(row.mean_delay_censored, row.detect_rate,
                        xerr=row.mean_delay_censored_se,
                        yerr=row.detect_rate_se, fmt="o", label=row.method)
            ax.annotate(row.method.replace("lsc_", "L:"),
                        (row.mean_delay_censored, row.detect_rate),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(scen, fontsize=9)
        ax.set_xlabel("mean delay (censored)")
    axes[0].set_ylabel("detect rate")
    fig.suptitle("exp01 AR(1) arena — calibrated FAR 5%/500 obs, 500 reps")
    fig.tight_layout()
    fig.savefig("paper_assets/exp01_frontier.png", dpi=130)
    print("\nwrote paper_assets/exp01_frontier.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "paper_assets/exp01_results.csv")
