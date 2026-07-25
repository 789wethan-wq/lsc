"""Assemble the phi=0.95 vs phi=0.99 robustness comparison (SPEC R2 M1).

Pulls estimated-rung rows from grid_v4_varbench_core (r, phi=0.95),
grid_v9_r_phi99 (r, phi=0.99), grid_v5_qbreak (q, phi=0.95, arena
ar1_snr2.0 -- the closest phi=0.95 SNR to the phi=0.99 q-channel's
fixed-q induced SNR ~2.45, NOT an exact SNR match, flagged explicitly
in the output), grid_v8_phiqbreak (q, phi=0.99, arena ar1_phi0.99);
known-parameter rows from exp26 (phi=0.95) and exp28 (phi=0.99).

Output: paper_assets/phi99_robustness_table.csv (+ printed table).
"""
from __future__ import annotations

import pandas as pd

A = "paper_assets"


def r_rows():
    rows = []
    d95 = pd.read_csv(f"{A}/grid_v4_varbench_core_results.csv")
    d99 = pd.read_csv(f"{A}/grid_v9_r_phi99_results.csv")
    for snr, arena95, arena99 in [(0.1, "ar1_snr0.1", "ar1_phi99_snr0.1"),
                                  (0.5, "ar1_snr0.5", "ar1_phi99_snr0.5"),
                                  (2.0, "ar1_snr2.0", "ar1_phi99_snr2.0")]:
        for vm, scen in [(1.5, "variance_x1.5"), (3.0, "variance_x3")]:
            def get(df, arena, method):
                m = df[(df.arena == arena) & (df.scenario == scen) & (df.method == method)]
                return float(m.detect_rate.iloc[0]) if len(m) else float("nan")

            rows.append(dict(channel="r", snr=snr, vol_mult=vm,
                             phi95_raw=get(d95, arena95, "raw_var_cusum"),
                             phi95_arima=get(d95, arena95, "arima_var_cusum"),
                             phi99_raw=get(d99, arena99, "raw_var_cusum"),
                             phi99_arima=get(d99, arena99, "arima_var_cusum")))
    return rows


def q_rows():
    rows = []
    d95 = pd.read_csv(f"{A}/grid_v5_qbreak_results.csv")
    d99 = pd.read_csv(f"{A}/grid_v8_phiqbreak_results.csv")
    for vm, scen95, scen99 in [(1.5, "qvar_x1.5", "qvar_x1.5"), (3.0, "qvar_x3", "qvar_x3")]:
        def get(df, arena, scen, method):
            m = df[(df.arena == arena) & (df.scenario == scen) & (df.method == method)]
            return float(m.detect_rate.iloc[0]) if len(m) else float("nan")

        rows.append(dict(channel="q", snr="2.0 (closest; phi99 induced SNR~2.45)", vol_mult=vm,
                         phi95_raw=get(d95, "ar1_snr2.0", scen95, "raw_var_cusum"),
                         phi95_arima=get(d95, "ar1_snr2.0", scen95, "arima_var_cusum"),
                         phi99_raw=get(d99, "ar1_phi0.99", scen99, "raw_var_cusum"),
                         phi99_arima=get(d99, "ar1_phi0.99", scen99, "arima_var_cusum")))
    return rows


def known_param_rows():
    k95 = pd.read_csv(f"{A}/exp26_known_param_variance.csv")
    k99 = pd.read_csv(f"{A}/exp28_known_param_phi99.csv")
    rows = []
    for _, r95 in k95.iterrows():
        # match phi99 row by channel + vol_mult; for q take the single phi99 cell
        # (induced SNR ~2.45) against every phi95 SNR row (labelled, not merged blindly)
        if r95.channel == "r":
            r99 = k99[(k99.channel == "r") & (k99.snr == r95.snr) & (k99.vol_mult == r95.vol_mult)]
        else:
            r99 = k99[(k99.channel == "q") & (k99.vol_mult == r95.vol_mult)]
        if len(r99) == 0:
            continue
        r99 = r99.iloc[0]
        rows.append(dict(channel=r95.channel, snr=r95.snr, vol_mult=r95.vol_mult,
                         phi95_known_raw=r95.detect_known_raw, phi95_known_kalman=r95.detect_known_kalman,
                         phi99_known_raw=r99.detect_known_raw, phi99_known_kalman=r99.detect_known_kalman))
    return rows


def main():
    est = pd.DataFrame(r_rows() + q_rows())
    known = pd.DataFrame(known_param_rows())
    est.to_csv(f"{A}/phi99_robustness_estimated.csv", index=False)
    known.to_csv(f"{A}/phi99_robustness_known.csv", index=False)
    print("=== estimated rungs, phi=0.95 vs phi=0.99 ===")
    print(est.round(3).to_string(index=False))
    print("\n=== known-parameter rungs, phi=0.95 vs phi=0.99 ===")
    print(known.round(3).to_string(index=False))
    print(f"\nwrote {A}/phi99_robustness_estimated.csv, {A}/phi99_robustness_known.csv")


if __name__ == "__main__":
    main()
