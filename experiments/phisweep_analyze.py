"""M3 analysis — does mu_inf sort innovation-CUSUM detection? (SPEC R1 §M3)

For each grid_v6 cell (phi, SNR, level magnitude) computes the predicted
asymptotic innovation drift mu_inf = delta(1-phi)/((1-phi(1-K)) sqrt(F))
(Proposition 1) and pairs it with the OBSERVED lsc_kalman_cusum
detection rate. mu_inf is increasing in (1-phi); the theory claims it
sorts detection, and the fast-or-never regime is mu_inf < k = 0.5 (the
CUSUM allowance). Produces the mu_inf-vs-detection scatter — the paper's
headline theory-verification figure — and a Spearman check that mu_inf
orders the detection rates.

Output: paper_assets/grid_v6_phisweep_muinf.csv (per-cell mu_inf +
detect), paper_assets/grid_v6_muinf_scatter.png.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from lsc.theory import mu_infinity, riccati_steady_state

A = "paper_assets"
K_ALLOW = 0.5  # innovation-CUSUM allowance (make_innovation_cusum_detector)
MAG = {"level_1s": 1.0, "level_3s": 3.0}


def main() -> None:
    cfg = yaml.safe_load(open("configs/grid_v6_phisweep.yaml"))
    arenas = cfg["arenas"]
    df = pd.read_parquet(f"{A}/grid_v6_phisweep_results.parquet")

    rows = []
    for _, r in df.iterrows():
        a = arenas[r.arena]
        phi, q, rr = float(a["phi"]), float(a["q"]), float(a["r"])
        _, K, F = riccati_steady_state(phi, q, rr)
        sigma_ref = np.sqrt(q / (1.0 - phi**2))
        snr = q / (1.0 - phi**2) / rr
        delta = MAG[r.scenario] * sigma_ref
        mu_inf = mu_infinity(delta, phi, K, F)
        rows.append(dict(arena=r.arena, phi=phi, snr=round(snr, 3),
                         scenario=r.scenario, magnitude=MAG[r.scenario],
                         method=r.method, mu_inf=mu_inf,
                         fast_regime=mu_inf >= K_ALLOW,
                         detect_rate=r.detect_rate,
                         detect_rate_se=r.detect_rate_se,
                         mean_delay=r.mean_delay_censored))
    out = pd.DataFrame(rows).sort_values(["scenario", "snr", "phi"])
    out.to_csv(f"{A}/grid_v6_phisweep_muinf.csv", index=False)

    kal = out[out.method == "lsc_kalman_cusum"]
    # Spearman: does mu_inf rank-order detection across all innovation cells?
    rho = kal[["mu_inf", "detect_rate"]].corr(method="spearman").iloc[0, 1]
    print("=== innovation CUSUM: mu_inf vs detect (sorted by mu_inf) ===")
    print(kal.sort_values("mu_inf")[
        ["phi", "snr", "scenario", "mu_inf", "detect_rate"]].round(3)
        .to_string(index=False))
    print(f"\nSpearman(mu_inf, detect_rate) = {rho:.3f}")
    print(f"fast-regime cells (mu_inf>={K_ALLOW}): "
          f"detect {kal[kal.fast_regime].detect_rate.min():.2f}"
          f"-{kal[kal.fast_regime].detect_rate.max():.2f}; "
          f"never-regime cells (mu_inf<{K_ALLOW}): detect "
          f"{kal[~kal.fast_regime].detect_rate.min():.2f}"
          f"-{kal[~kal.fast_regime].detect_rate.max():.2f}")

    _figure(out)
    print(f"\nwrote {A}/grid_v6_phisweep_muinf.csv, {A}/grid_v6_muinf_scatter.png")


def _figure(out: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kal = out[out.method == "lsc_kalman_cusum"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    markers = {"level_1s": "o", "level_3s": "s"}
    snr_c = {0.1: "C0", 0.5: "C1", 2.0: "C2"}
    for scen, gm in kal.groupby("scenario"):
        for snr, g in gm.groupby("snr"):
            axes[0].errorbar(g.mu_inf, g.detect_rate, yerr=g.detect_rate_se,
                             fmt=markers[scen], color=snr_c[round(snr, 1)],
                             ms=7, capsize=3,
                             label=f"{scen.replace('level_','')} SNR{snr}")
            # annotate phi at each point
            for _, r in g.iterrows():
                axes[0].annotate(f"{r.phi:g}", (r.mu_inf, r.detect_rate),
                                 fontsize=6, xytext=(3, 3),
                                 textcoords="offset points")
    axes[0].axvline(K_ALLOW, color="k", ls=":", lw=1,
                    label=f"fast-or-never boundary k={K_ALLOW}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"predicted $\mu_\infty$ (log scale)")
    axes[0].set_ylabel("innovation-CUSUM detect rate")
    axes[0].set_title(r"(a) $\mu_\infty$ sorts detection (labels = $\phi$)",
                      fontsize=9)
    axes[0].legend(fontsize=6, ncol=2)
    # (b) detection vs phi, showing the escape at low phi (3-sigma)
    for snr, g in kal[kal.scenario == "level_3s"].groupby("snr"):
        g = g.sort_values("phi")
        axes[1].plot(g.phi, g.detect_rate, "o-", color=snr_c[round(snr, 1)],
                     label=f"SNR {snr}")
    axes[1].set_xlabel(r"$\phi$")
    axes[1].set_ylabel("detect rate (3σ level)")
    axes[1].set_title(r"(b) innovation CUSUM escapes fast-or-never at low $\phi$",
                      fontsize=9)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{A}/grid_v6_muinf_scatter.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
