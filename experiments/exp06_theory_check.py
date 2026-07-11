"""exp06 — numerical verification of the fast-or-never theory
(experiments/THEORY.md; predictions pre-registered in CHANGELOG,
2026-07-11 exp05/exp06 entry).

Three checks, all with the steady-state KNOWN-parameter filter the
theory describes (lsc.theory.steady_state_innovations):

A. Innovation mean path: MC average of standardized innovations after
   a 3 sigma_ref state level shift vs the deterministic path mu_t of
   Proposition 1 (SNR 0.5 arena).
B. Detection probability vs threshold h: full-filter MC vs the reduced
   simulation (iid N(0,1) + mu_t, Proposition 1 reduction) vs the
   finite-horizon never-detect bound of Proposition 2, at delta = 1 and
   3 sigma_ref.
C. The mu_inf table across the grid_v1 arenas (SNR x delta), with the
   bound evaluated at each arena's actually-calibrated innovation-CUSUM
   threshold, next to grid_v1's observed detect rates; plus Wald delay
   (Proposition 3) vs raw CUSUM's observed median delay. grid_v1
   detectors use FITTED (training-prefix) parameters and a two-sided
   CUSUM, so this comparison is a known-parameter approximation to the
   experiment — documented in THEORY.md.

Outputs: paper_assets/exp06_theory_table.csv,
exp06_innovation_path.png, exp06_detect_vs_h.png.

Usage: python experiments/exp06_theory_check.py [n_reps]
"""
from __future__ import annotations

import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.theory import (
    innovation_mean_path,
    mu_infinity,
    never_detect_bound,
    riccati_steady_state,
    steady_state_innovations,
    wald_delay,
)

PHI, R = 0.95, 1.0
K_ALLOW = 0.5
T, T0 = 500, 250
SEED0 = 600_000  # disjoint from all calibration/evaluation ranges


def one_sided_cusum_max(e: np.ndarray, k: float) -> float:
    g, gmax = 0.0, 0.0
    for x in e:
        g = max(0.0, g + x - k)
        gmax = max(gmax, g)
    return gmax


def arena(snr: float) -> AR1StateDGP:
    return AR1StateDGP(phi=PHI, q=snr * (1 - PHI**2), r=R)


def part_a(n_reps: int) -> None:
    snr = 0.5
    dgp = arena(snr)
    q = dgp.q
    dgp_b = AR1StateDGP(phi=PHI, q=q, r=R, breaks=[
        BreakSpec(kind="level", time_frac=0.5, magnitude=3.0)])
    acc = np.zeros(T - T0)
    for i in range(n_reps):
        Y = dgp_b.sample(T, seed=SEED0 + i).Y
        acc += steady_state_innovations(Y, PHI, q, R)[T0:]
    emp = acc / n_reps
    _, K, F = riccati_steady_state(PHI, q, R)
    delta = 3.0 * dgp.sigma_ref
    mu = innovation_mean_path(delta, PHI, K, F, T - T0)
    se = 1.0 / np.sqrt(n_reps)
    print(f"A. innovation mean path: max|MC - theory| = "
          f"{np.max(np.abs(emp - mu)):.4f}  (per-point MC SE {se:.4f})")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(T0, T), emp, lw=0.8, label=f"MC mean ({n_reps} reps)")
    ax.plot(np.arange(T0, T), mu, "r--", lw=1.5,
            label=r"theory $\mu_t$ (Prop. 1)")
    ax.axhline(K_ALLOW, color="gray", ls=":", label=f"allowance k={K_ALLOW}")
    ax.axhline(mu_infinity(delta, PHI, K, F), color="r", ls=":",
               label=r"$\mu_\infty$ = "
                     f"{mu_infinity(delta, PHI, K, F):.3f}")
    ax.set_xlabel("t (break at 250)")
    ax.set_ylabel("standardized innovation mean")
    ax.set_title("Post-break innovation mean: filter MC vs Proposition 1 "
                 f"(SNR 0.5, 3$\\sigma$ level shift); band = ±2 MC SE")
    ax.fill_between(np.arange(T0, T), mu - 2 * se, mu + 2 * se,
                    color="r", alpha=0.15)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("paper_assets/exp06_innovation_path.png", dpi=130)
    plt.close(fig)


def part_b(n_reps: int) -> None:
    snr = 0.5
    q = snr * (1 - PHI**2)
    _, K, F = riccati_steady_state(PHI, q, R)
    sigma_ref = np.sqrt(q / (1 - PHI**2))
    h_grid = np.linspace(4, 40, 19)
    L = T - T0
    rng = np.random.default_rng(SEED0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, mag in zip(axes, (1.0, 3.0)):
        delta = mag * sigma_ref
        mu = innovation_mean_path(delta, PHI, K, F, L)
        mu_inf = mu_infinity(delta, PHI, K, F)
        dgp_b = AR1StateDGP(phi=PHI, q=q, r=R, breaks=[
            BreakSpec(kind="level", time_frac=0.5, magnitude=mag)])
        full = np.array([one_sided_cusum_max(
            steady_state_innovations(
                dgp_b.sample(T, seed=SEED0 + 10_000 + i).Y,
                PHI, q, R)[T0:], K_ALLOW)
            for i in range(n_reps)])
        reduced = np.array([one_sided_cusum_max(
            rng.standard_normal(L) + mu, K_ALLOW) for _ in range(n_reps)])
        p_full = [(full >= h).mean() for h in h_grid]
        p_red = [(reduced >= h).mean() for h in h_grid]
        p_bound = [never_detect_bound(mu_inf, K_ALLOW, h, L) for h in h_grid]
        ax.plot(h_grid, p_full, "o-", ms=3, label="full filter MC")
        ax.plot(h_grid, p_red, "s--", ms=3,
                label=r"reduced: iid N(0,1)+$\mu_t$")
        ax.plot(h_grid, p_bound, "r:", label="Prop. 2 bound")
        ax.set_title(rf"$\delta$={mag:g}$\sigma_{{ref}}$, "
                     rf"$\mu_\infty$={mu_inf:.3f} vs k={K_ALLOW}")
        ax.set_xlabel("threshold h")
        se = np.sqrt(0.25 / n_reps)
        print(f"B. delta={mag:g}s: max|full-reduced| = "
              f"{np.max(np.abs(np.array(p_full) - np.array(p_red))):.4f} "
              f"(2*MC SE ~ {2*se:.4f}); "
              f"bound violated: {any(np.array(p_full) > np.array(p_bound) + 3*se)}")
    axes[0].set_ylabel("P(alarm after break)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Detection probability vs threshold: fast-or-never "
                 "(SNR 0.5, one-sided CUSUM, k=0.5, 250 post-break obs)")
    fig.tight_layout()
    fig.savefig("paper_assets/exp06_detect_vs_h.png", dpi=130)
    plt.close(fig)


def part_c() -> pd.DataFrame:
    far = pd.read_csv("paper_assets/grid_v1_far_calibration.csv")
    res = pd.read_csv("paper_assets/grid_v1_results.csv")
    rows = []
    for snr_name in ("0.1", "0.5", "2.0"):
        snr = float(snr_name)
        aname = f"ar1_snr{snr_name}"
        q = snr * (1 - PHI**2)
        _, K, F = riccati_steady_state(PHI, q, R)
        sigma_ref = np.sqrt(q / (1 - PHI**2))
        sigma_y = np.sqrt(sigma_ref**2 + R)
        h_lsc = float(far[(far.arena == aname)
                          & (far.method == "lsc_kalman_cusum")].threshold.iloc[0])
        h_raw = float(far[(far.arena == aname)
                          & (far.method == "raw_cusum")].threshold.iloc[0])
        for mag, scen in ((0.5, "level_0.5s"), (1.0, "level_1s"),
                          (3.0, "level_3s")):
            delta = mag * sigma_ref
            mu_inf = mu_infinity(delta, PHI, K, F)
            sub = res[(res.arena == aname) & (res.scenario == scen)]
            obs_lsc = float(sub[sub.method == "lsc_kalman_cusum"].detect_rate.iloc[0])
            obs_raw = float(sub[sub.method == "raw_cusum"].detect_rate.iloc[0])
            obs_raw_delay = float(
                sub[sub.method == "raw_cusum"].median_delay_detected.iloc[0])
            shift_std = delta / sigma_y
            rows.append(dict(
                snr=snr, magnitude=mag, K=round(K, 3),
                sqrtF=round(np.sqrt(F), 3), mu_inf=round(mu_inf, 3),
                gap_k_minus_mu=round(K_ALLOW - mu_inf, 3),
                bound_at_h=round(never_detect_bound(
                    mu_inf, K_ALLOW, h_lsc, 250), 4),
                h_lsc=round(h_lsc, 1),
                obs_lsc_detect=obs_lsc,
                raw_shift_std=round(shift_std, 3),
                wald_raw_delay=round(wald_delay(shift_std, K_ALLOW, h_raw), 1),
                obs_raw_median_delay=obs_raw_delay,
                obs_raw_detect=obs_raw,
            ))
    df = pd.DataFrame(rows)
    df.to_csv("paper_assets/exp06_theory_table.csv", index=False)
    print("C. mu_inf / bound table vs grid_v1 observations:")
    print(df.to_string(index=False))
    return df


def main(n_reps: int = 1000) -> None:
    t0 = time.time()
    part_a(n_reps)
    part_b(n_reps)
    part_c()
    print(f"[{time.time()-t0:.0f}s] wrote paper_assets/exp06_*")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
