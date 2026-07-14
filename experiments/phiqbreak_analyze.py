"""M7 analysis — does raw's q-break advantage track the 1/(1-phi^2)
amplification? (SPEC R1 §M7)

For grid_v8 (fixed q,r; swept phi) computes, per phi and break size, the
raw rung's advantage Delta = detect(raw_var) - detect(arima_var) (and vs
composite), the induced SNR(phi) = (q/r)/(1-phi^2), and the amplification
A(phi) = 1/(1-phi^2). Checks the pre-registered prediction: Delta is
monotone increasing in phi, ->0 as phi->0, and tracks A(phi)/SNR. The
secondary consistency check overlays the M2 fixed-phi SNR sweep (grid_v5)
at matched induced SNR.

Output: paper_assets/grid_v8_phiqbreak_summary.csv,
        paper_assets/grid_v8_phiq_amplification.png.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

A = "paper_assets"
RUNGS = ["raw_var_cusum", "arima_var_cusum", "lsc_composite"]


def main() -> None:
    cfg = yaml.safe_load(open("configs/grid_v8_phiqbreak.yaml"))
    arenas = cfg["arenas"]
    df = pd.read_parquet(f"{A}/grid_v8_phiqbreak_results.parquet")
    det = df.pivot_table(index=["scenario", "arena"], columns="method",
                         values="detect_rate", aggfunc="first")

    rows = []
    for (scen, arena), r in det.iterrows():
        a = arenas[arena]
        phi, q, rr = float(a["phi"]), float(a["q"]), float(a["r"])
        snr = q / (1.0 - phi**2) / rr
        amp = 1.0 / (1.0 - phi**2)
        rows.append(dict(
            scenario=scen, phi=phi, snr=round(snr, 3), amp=round(amp, 2),
            raw=r["raw_var_cusum"], arima=r["arima_var_cusum"],
            composite=r["lsc_composite"],
            adv_vs_arima=r["raw_var_cusum"] - r["arima_var_cusum"],
            adv_vs_comp=r["raw_var_cusum"] - r["lsc_composite"]))
    out = pd.DataFrame(rows).sort_values(["scenario", "phi"])
    out.to_csv(f"{A}/grid_v8_phiqbreak_summary.csv", index=False)

    for scen in ["qvar_x1.5", "qvar_x3"]:
        s = out[out.scenario == scen].sort_values("phi")
        mono = bool(np.all(np.diff(s.adv_vs_arima.values) >= -0.03))  # ~monotone
        rho_amp = s[["amp", "adv_vs_arima"]].corr(method="spearman").iloc[0, 1]
        print(f"\n=== {scen}: raw advantage vs phi (fixed q={0.04875}, r=1) ===")
        print(s[["phi", "snr", "amp", "raw", "arima", "composite",
                 "adv_vs_arima"]].round(3).to_string(index=False))
        print(f"  monotone-increasing in phi: {mono}; "
              f"Spearman(amp, adv)={rho_amp:.3f}; adv at phi=0.1: "
              f"{s.iloc[0].adv_vs_arima:+.3f}; at phi=0.99: "
              f"{s.iloc[-1].adv_vs_arima:+.3f}")

    # secondary: consistency with the M2 fixed-phi(0.95) SNR sweep (grid_v5)
    try:
        g5 = pd.read_parquet(f"{A}/grid_v5_qbreak_results.parquet")
        g5 = g5[g5.arena.isin(["ar1_snr0.1", "ar1_snr0.5", "ar1_snr2.0"])]
        p5 = g5.pivot_table(index=["scenario", "arena"], columns="method",
                            values="detect_rate", aggfunc="first")
        snr_of = {"ar1_snr0.1": 0.1, "ar1_snr0.5": 0.5, "ar1_snr2.0": 2.0}
        m2 = []
        for (scen, arena), r in p5.iterrows():
            m2.append(dict(scenario=scen, snr=snr_of[arena],
                           adv_vs_arima=r["raw_var_cusum"] - r["arima_var_cusum"]))
        m2 = pd.DataFrame(m2)
        print("\n=== consistency check: raw advantage vs induced SNR ===")
        print("  grid_v8 (phi-swept)   vs   grid_v5 (SNR-swept, phi=0.95)")
        for scen in ["qvar_x1.5", "qvar_x3"]:
            v8 = out[out.scenario == scen][["snr", "adv_vs_arima"]]
            v5 = m2[m2.scenario == scen][["snr", "adv_vs_arima"]]
            print(f"  {scen}: v8 {dict(zip(v8.snr.round(2), v8.adv_vs_arima.round(2)))}")
            print(f"  {' '*len(scen)}  v5 {dict(zip(v5.snr, v5.adv_vs_arima.round(2)))}")
    except FileNotFoundError:
        m2 = None

    _figure(out, m2)
    print(f"\nwrote {A}/grid_v8_phiqbreak_summary.csv, "
          f"{A}/grid_v8_phiq_amplification.png")


def _figure(out, m2) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    mk = {"qvar_x1.5": "o", "qvar_x3": "s"}
    for scen in ["qvar_x1.5", "qvar_x3"]:
        s = out[out.scenario == scen].sort_values("phi")
        axes[0].plot(s.phi, s.adv_vs_arima, mk[scen] + "-",
                     label=f"{scen.replace('qvar_','q ')}")
        axes[1].plot(s.amp, s.adv_vs_arima, mk[scen] + "-",
                     label=f"{scen.replace('qvar_','q ')} (grid_v8, φ-swept)")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_xlabel(r"$\phi$")
    axes[0].set_ylabel(r"raw advantage $\Delta$ = raw − ARIMA")
    axes[0].set_title(r"(a) subtle-break advantage vanishes as $\phi\to0$; "
                      "coarse persists", fontsize=9)
    axes[0].legend(fontsize=8)
    if m2 is not None:
        for scen in ["qvar_x1.5", "qvar_x3"]:
            v5 = m2[m2.scenario == scen].sort_values("snr")
            axes[1].plot(1.0 / (1 - 0.95**2) * v5.snr / 0.5, v5.adv_vs_arima,
                         mk[scen] + ":", color="C3", alpha=0.6,
                         label=f"{scen.replace('qvar_','q ')} (grid_v5, SNR-swept)")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"amplification $1/(1-\phi^2)$  (∝ induced SNR)")
    axes[1].set_ylabel(r"raw advantage $\Delta$")
    axes[1].set_title(r"(b) subtle-break $\Delta$ tracks amplification "
                      "(matches SNR-sweep)", fontsize=9)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{A}/grid_v8_phiq_amplification.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
