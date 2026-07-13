"""Assemble the two-channel whitening-ladder table (SPEC R1 §M2).

Extends `paper_assets/ladder_table.csv` with a `break_channel` column
(r | q) and the q-break rows from grid_v5. The r-channel rows are the
observation-noise ('variance') results already assembled by
`varbench_ladder.py`; the q-channel rows are the state-innovation
('state_var') results from grid_v5_qbreak, run with identical seed
blocks so the two channels are draw-for-draw comparable at each ladder
rung. The magnitude label is normalized (x1.5 / x3 / x0.67) so the r and
q rows line up in the paper's §5 table.

Output: paper_assets/ladder_table.csv (now two-channel, break_channel
column) + printed r-vs-q pivot at each SNR arena.
"""
from __future__ import annotations

import pandas as pd

A = "paper_assets"
RUNGS = ["raw_var_cusum", "arima_var_cusum", "lsc_composite", "lsc_tail_cusum"]


def _mag(scenario: str) -> str:
    # variance_x1.5 / qvar_x1.5 -> x1.5 ; variance_x0.67 / qvar_x0.67 -> x0.67
    return scenario.split("_", 1)[1]


def main() -> None:
    # r-channel: reuse the varbench ladder assembly (rebuild if absent)
    try:
        rtab = pd.read_csv(f"{A}/ladder_table.csv")
    except FileNotFoundError:
        import varbench_ladder
        varbench_ladder.main()
        rtab = pd.read_csv(f"{A}/ladder_table.csv")
    rtab = rtab[rtab.scenario.str.startswith("variance_")].copy()
    # if a previous 2-channel table is re-read, drop its channel col first
    rtab = rtab.drop(columns=[c for c in ("break_channel", "mag") if c in rtab])
    rtab["break_channel"] = "r"

    # q-channel: grid_v5
    q = pd.read_parquet(f"{A}/grid_v5_qbreak_results.parquet")
    q = q[q.scenario.str.startswith("qvar_")].copy()
    q = q[["T", "arena", "scenario", "method", "detect_rate",
           "detect_rate_se", "mean_delay_censored"]]
    q["source"] = "grid_v5_qbreak"
    q["break_channel"] = "q"

    both = pd.concat([rtab, q], ignore_index=True)
    both["mag"] = both.scenario.map(_mag)
    both = (both[["T", "arena", "break_channel", "mag", "scenario", "method",
                  "detect_rate", "detect_rate_se", "mean_delay_censored",
                  "source"]]
            .drop_duplicates(["T", "arena", "break_channel", "mag", "method"])
            .sort_values(["arena", "break_channel", "mag", "method"]))
    both.to_csv(f"{A}/ladder_table.csv", index=False)

    # printed comparison: rung detect rates, r vs q, at x1.5 T=500
    for mag in ["x1.5", "x3", "x0.67"]:
        sub = both[(both["T"] == 500) & (both["mag"] == mag)
                   & both.arena.isin(["ar1_snr0.1", "ar1_snr0.5", "ar1_snr2.0"])]
        piv = sub.pivot_table(index=["break_channel", "method"],
                              columns="arena", values="detect_rate",
                              aggfunc="first")
        cols = [c for c in ["ar1_snr0.1", "ar1_snr0.5", "ar1_snr2.0"]
                if c in piv.columns]
        print(f"\n=== {mag}  (detect rate by SNR) ===")
        print(piv[cols].reindex(
            pd.MultiIndex.from_product([["r", "q"], RUNGS])).round(3).to_string())

    far = pd.read_csv(f"{A}/grid_v5_qbreak_far_calibration.csv")
    print("\n=== q-break empirical FARs (target 0.05) ===")
    print(far.pivot_table(index="arena", columns="method",
                          values="far_empirical", aggfunc="first")
          [RUNGS].round(3).to_string())
    print(f"\nwrote {A}/ladder_table.csv (two-channel, break_channel r|q)")


if __name__ == "__main__":
    main()
