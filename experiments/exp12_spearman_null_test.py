"""exp12 -- real n and a stratified permutation null for the paper's
"mu_inf sorts detection, Spearman 0.94" claim (referee-requested,
external review this round). The paper reports this correlation
without stating n or testing it against a null where a nuisance
dimension (SNR, shift size) rather than mu_inf/phi specifically could
be driving the ordering.

Uses the REAL, already-computed, estimated-parameter grid_v6_phisweep
output (paper_assets/grid_v6_phisweep_muinf.csv, produced by
experiments/phisweep_analyze.py from the pinned grid_v6_phisweep run)
directly -- not a from-scratch simulation -- so n and the correlation
here are exactly what the paper's own pipeline produced.

configs/grid_v6_phisweep.yaml has 4 phi values x 3 SNR arenas x 2
level scenarios (level_1s, level_3s) for lsc_kalman_cusum -- n = 24,
not the 4x3x3=36 an outside reviewer guessed without the config.

Stratified permutation test: within each of the 6 (SNR, scenario)
strata (4 phi-values each), shuffle which mu_inf is paired with which
detect_rate, recompute the overall Spearman, repeat n_perm times. If
mu_inf's sorting power depended only on SNR/shift (not phi specifically,
as the theory claims), the observed Spearman should sit inside this
permutation null; if it is really about phi/mu_inf, the observed value
should sit far in the tail.

Usage: python experiments/exp12_spearman_null_test.py [n_perm]
Output: paper_assets/exp12_spearman_null_real.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEED = 20260720


def main(n_perm: int = 20000) -> None:
    df = pd.read_csv("paper_assets/grid_v6_phisweep_muinf.csv")
    kal = df[df.method == "lsc_kalman_cusum"].reset_index(drop=True)
    n = len(kal)
    print(f"n = {n} cells (grid_v6_phisweep.yaml: 4 phi x 3 SNR x "
          f"{kal.scenario.nunique()} scenarios, lsc_kalman_cusum only)")

    mu_vals = kal["mu_inf"].to_numpy()
    det_vals = kal["detect_rate"].to_numpy()
    rho_obs, p_asymp = spearmanr(mu_vals, det_vals)
    print(f"Observed Spearman(mu_inf, detect_rate) = {rho_obs:.4f} "
          f"(asymptotic p={p_asymp:.2e})")

    strata: dict[tuple, list[int]] = {}
    for i, r in kal.iterrows():
        strata.setdefault((r["snr"], r["scenario"]), []).append(i)
    print(f"strata: {len(strata)} groups of "
          f"{sorted(set(len(v) for v in strata.values()))} phi-values each")

    rng = np.random.default_rng(SEED)
    null_rhos = np.empty(n_perm)
    for p in range(n_perm):
        mu_shuffled = mu_vals.copy()
        for idxs in strata.values():
            idxs = np.array(idxs)
            mu_shuffled[idxs] = mu_vals[rng.permutation(idxs)]
        rho, _ = spearmanr(mu_shuffled, det_vals)
        null_rhos[p] = rho

    perm_p = float((np.abs(null_rhos) >= abs(rho_obs)).mean())
    print(f"Stratified permutation null (shuffle phi-pairing within "
          f"SNR x scenario, n_perm={n_perm}):")
    print(f"  null Spearman mean={null_rhos.mean():.4f}, "
          f"sd={null_rhos.std():.4f}, 95th pct={np.quantile(null_rhos, 0.95):.4f}, "
          f"max={null_rhos.max():.4f}")
    print(f"  permutation p-value (two-sided) = {perm_p:.5f}")

    out = kal[["arena", "phi", "snr", "scenario", "mu_inf", "detect_rate"]].copy()
    out.to_csv("paper_assets/exp12_spearman_null_real.csv", index=False)
    summary = pd.DataFrame([dict(
        n=n, rho_obs=round(float(rho_obs), 4), p_asymptotic=float(p_asymp),
        n_perm=n_perm, null_mean=round(float(null_rhos.mean()), 4),
        null_sd=round(float(null_rhos.std()), 4),
        null_95pct=round(float(np.quantile(null_rhos, 0.95)), 4),
        null_max=round(float(null_rhos.max()), 4),
        perm_p_two_sided=perm_p)])
    summary.to_csv("paper_assets/exp12_spearman_null_summary.csv", index=False)
    pd.DataFrame({"null_rho": null_rhos}).to_csv(
        "paper_assets/exp12_spearman_null_raw.csv", index=False)
    print("\nwrote paper_assets/exp12_spearman_null_real.csv, "
          "paper_assets/exp12_spearman_null_summary.csv, "
          "paper_assets/exp12_spearman_null_raw.csv")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
