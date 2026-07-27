"""r-channel phi-sweep assembler (SPEC R5 M1): combines the NEW
phi in {0.5, 0.8} cells (grid_v9b_r_phi_lo) with the EXISTING
phi=0.95 (grid_v4_varbench_core) and phi=0.99 (grid_v9_r_phi99,
R2 this round) r-channel results into the full 4x2x3 (phi x vol_mult x
SNR) table Table 3b's single phi=0.99 point was missing, matching
Table 4/Fig. 2's level-shift phi-sweep in scope. Same amplification-
factor summary convention as phiqbreak_analyze.py (Table 4's own
assembler): Spearman(1/(1-phi^2), raw-minus-arima advantage) per break
size.

Output: paper_assets/r_phi_sweep_full.csv,
        paper_assets/r_phi_sweep_amplification.png.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

A = "paper_assets"

# (phi, results_path, arena_name, snr) -- q values match grid_v6_phisweep's
# convention (q = SNR*(1-phi^2)*r) throughout, so all four phi rows are
# draw-for-draw comparable to the existing level-shift phi-sweep at the
# same (phi, SNR).
SOURCES = [
    (0.5, f"{A}/grid_v9b_r_phi_lo_results.csv", "ar1_phi0.5_snr{snr}"),
    (0.8, f"{A}/grid_v9b_r_phi_lo_results.csv", "ar1_phi0.8_snr{snr}"),
    (0.95, f"{A}/grid_v4_varbench_core_results.csv", "ar1_snr{snr}"),
    (0.99, f"{A}/grid_v9_r_phi99_results.csv", "ar1_phi99_snr{snr}"),
]
SNRS = (0.1, 0.5, 2.0)
SCEN_MAP = {1.5: "variance_x1.5", 3.0: "variance_x3"}


def _rate(path: str, arena: str, scenario: str, method: str) -> float:
    df = pd.read_csv(path)
    m = df[(df.arena == arena) & (df.scenario == scenario) & (df.method == method)]
    return float(m.detect_rate.iloc[0]) if len(m) else float("nan")


def main() -> None:
    rows = []
    for phi, path, arena_fmt in SOURCES:
        for snr in SNRS:
            arena = arena_fmt.format(snr=snr)
            for vol_mult, scen in SCEN_MAP.items():
                raw = _rate(path, arena, scen, "raw_var_cusum")
                arima = _rate(path, arena, scen, "arima_var_cusum")
                rows.append(dict(phi=phi, snr=snr, vol_mult=vol_mult,
                                 amp=1.0 / (1.0 - phi**2),
                                 raw=raw, arima=arima, adv_vs_arima=raw - arima))
    out = pd.DataFrame(rows).sort_values(["vol_mult", "snr", "phi"]).reset_index(drop=True)
    out.to_csv(f"{A}/r_phi_sweep_full.csv", index=False)

    print("=== full 4x2x3 r-channel phi-sweep table ===")
    print(out.round(3).to_string(index=False))

    for vol_mult in (1.5, 3.0):
        for snr in SNRS:
            s = out[(out.vol_mult == vol_mult) & (out.snr == snr)].sort_values("phi")
            rho_amp = s[["amp", "adv_vs_arima"]].corr(method="spearman").iloc[0, 1]
            print(f"\nx{vol_mult} SNR={snr}: adv_vs_arima by phi = "
                  f"{dict(zip(s.phi, s.adv_vs_arima.round(3)))}, "
                  f"Spearman(amp, adv)={rho_amp:.3f}")

    _figure(out)
    print(f"\nwrote {A}/r_phi_sweep_full.csv, {A}/r_phi_sweep_amplification.png")


def _figure(out: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    mk = {1.5: "o", 3.0: "s"}
    for i, snr in enumerate(SNRS):
        ax = axes[i]
        for vol_mult in (1.5, 3.0):
            s = out[(out.vol_mult == vol_mult) & (out.snr == snr)].sort_values("phi")
            ax.plot(s.phi, s.raw, mk[vol_mult] + "-", color="C0", label=f"raw x{vol_mult}")
            ax.plot(s.phi, s.arima, mk[vol_mult] + "--", color="C1", label=f"ARIMA x{vol_mult}")
        ax.set_xlabel(r"$\phi$")
        ax.set_title(f"SNR={snr}", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.set_ylabel("detect_rate")
            ax.legend(fontsize=7)
    fig.suptitle("r-channel whitening ladder across the full phi-sweep", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{A}/r_phi_sweep_amplification.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
