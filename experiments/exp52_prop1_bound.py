"""exp52 -- numerical evaluation of the Proposition 1(b) / Proposition 2
never-detect bound (SPEC_R8_missing_experiments.md S10; pre-registered
in experiments/CHANGELOG.md 2026-07-27 BEFORE this script was run).

Pure computation, no simulation. Sec 4 currently cites "the Proposition
2 bound is never violated" as a verification of the theory; a bound
that has already saturated at 1.0 in the published exp06_theory_table
(magnitude=3.0 rows, all three SNRs) cannot be violated by
construction and is not a verification of anything. This computes the
bound with the EXACT Proposition 2 formula from THEORY.md --

    (L+1) * exp(-2 (k - mu_inf) (h - g)),  g = 0

-- reported UNCAPPED (not min(1, ...)), across grid_v1 (phi=0.95, the
three published SNRs x three published magnitudes) and
grid_v6_phisweep (phi in {0.5,0.8,0.95,0.99}, same SNRs, magnitudes
{1,3}), at both L=250 (post-break horizon, T - break_time) and L=375
(T - n_train, the full monitored window).

lsc.theory.never_detect_bound implements a related but NOT identical
formula (L * exp(...), no +1, implicitly g=0, and separately caps at
1.0 internally) -- this script deliberately does not reuse it, so the
(L+1) vs L / capped-vs-raw discrepancy is visible rather than hidden
inside a shared helper.

h is read from each arena's OWN published lsc_kalman_cusum threshold
(<config>_far_calibration.csv) -- the actual calibrated threshold, not
a nominal one. observed_detect_est is the matching published detect
rate; observed_detect_known is populated only at the single cell
exp10_cusum_ablation.csv covers (phi=0.95, SNR=0.5, magnitude=3.0,
two-sided/known variant) and is NaN everywhere else -- reported as
such, not backfilled.

Usage: python experiments/exp52_prop1_bound.py
Output: paper_assets/exp52_prop1_bound.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lsc.theory import mu_infinity, riccati_steady_state

REPO_ROOT = Path(__file__).resolve().parent.parent
A = REPO_ROOT / "paper_assets"
OUT_PATH = A / "exp52_prop1_bound.csv"

K_ALLOW = 0.5
R = 1.0
T, N_TRAIN, T0 = 500, 125, 250
L_POST_BREAK = T - T0          # 250
L_FULL_WINDOW = T - N_TRAIN    # 375


def bound_value(mu_inf: float, k: float, h: float, L: int, g: float = 0.0) -> float:
    """Proposition 2, exact form, uncapped: (L+1) exp(-2(k-mu)(h-g))."""
    return float((L + 1) * np.exp(-2.0 * (k - mu_inf) * (h - g)))


def known_detect_rate(phi: float, snr: float, magnitude: float) -> float:
    if not (np.isclose(phi, 0.95) and np.isclose(snr, 0.5) and np.isclose(magnitude, 3.0)):
        return float("nan")
    df = pd.read_csv(A / "exp10_cusum_ablation.csv")
    row = df[df.variant == "c_two_sided_known"]
    return float(row.detect_rate.iloc[0]) if len(row) else float("nan")


def cells_grid_v1() -> list[dict]:
    far = pd.read_csv(A / "grid_v1_far_calibration.csv")
    res = pd.read_csv(A / "grid_v1_results.csv")
    PHI = 0.95
    out = []
    for snr_name, scen in [("0.1", "level_0.5s"), ("0.1", "level_1s"), ("0.1", "level_3s"),
                            ("0.5", "level_0.5s"), ("0.5", "level_1s"), ("0.5", "level_3s"),
                            ("2.0", "level_0.5s"), ("2.0", "level_1s"), ("2.0", "level_3s")]:
        snr = float(snr_name)
        arena = f"ar1_snr{snr_name}"
        magnitude = {"level_0.5s": 0.5, "level_1s": 1.0, "level_3s": 3.0}[scen]
        h_row = far[(far.arena == arena) & (far.method == "lsc_kalman_cusum")]
        if h_row.empty:
            continue
        h = float(h_row.threshold.iloc[0])
        obs_row = res[(res.arena == arena) & (res.scenario == scen)
                      & (res.method == "lsc_kalman_cusum")]
        obs_est = float(obs_row.detect_rate.iloc[0]) if len(obs_row) else float("nan")
        out.append(dict(config="grid_v1", phi=PHI, snr=snr, magnitude=magnitude,
                        arena=arena, scenario=scen, h=h, observed_detect_est=obs_est))
    return out


def cells_grid_v6() -> list[dict]:
    far = pd.read_csv(A / "grid_v6_phisweep_far_calibration.csv")
    res = pd.read_csv(A / "grid_v6_phisweep_results.csv")
    out = []
    snr_tags = {0.1: "0.1", 0.5: "0.5", 2.0: "2.0"}
    for phi in (0.5, 0.8, 0.95, 0.99):
        for snr in (0.1, 0.5, 2.0):
            phi_tag = f"{phi:g}"
            snr_tag = snr_tags[snr]
            arena = f"ar1_phi{phi_tag}_snr{snr_tag}"
            for scen, magnitude in (("level_1s", 1.0), ("level_3s", 3.0)):
                h_row = far[(far.arena == arena) & (far.method == "lsc_kalman_cusum")]
                if h_row.empty:
                    continue
                h = float(h_row.threshold.iloc[0])
                obs_row = res[(res.arena == arena) & (res.scenario == scen)
                              & (res.method == "lsc_kalman_cusum")]
                obs_est = float(obs_row.detect_rate.iloc[0]) if len(obs_row) else float("nan")
                out.append(dict(config="grid_v6_phisweep", phi=phi, snr=snr,
                                magnitude=magnitude, arena=arena, scenario=scen,
                                h=h, observed_detect_est=obs_est))
    return out


N_REPS_RESOLUTION = 500          # grid_v1/grid_v6's own evaluation n_reps
RESOLUTION_FLOOR = 1.0 / N_REPS_RESOLUTION   # ~0.002: below this, no feasible
                                              # simulation at this grid's budget
                                              # could ever observe a violation


def main() -> pd.DataFrame:
    cells = cells_grid_v1() + cells_grid_v6()

    # Reference threshold per (snr, magnitude): the phi=0.95 (grid_v1) arena's
    # OWN calibrated h at that cell -- held FIXED while phi varies, to isolate
    # the mu_inf-only effect from the threshold-growth (near-unit-root
    # calibration) effect that otherwise dominates the raw bound comparison.
    h_ref = {(c["snr"], c["magnitude"]): c["h"] for c in cells if c["config"] == "grid_v1"}

    rows = []
    for c in cells:
        phi, snr, magnitude, h = c["phi"], c["snr"], c["magnitude"], c["h"]
        q = snr * (1.0 - phi**2) * R
        sigma_ref = float(np.sqrt(q / (1.0 - phi**2)))
        delta = magnitude * sigma_ref
        P, K, F = riccati_steady_state(phi, q, R)
        mu_inf = mu_infinity(delta, phi, K, F)
        obs_known = known_detect_rate(phi, snr, magnitude)
        h_fixed = h_ref[(snr, magnitude)]
        for L in (L_POST_BREAK, L_FULL_WINDOW):
            bv = bound_value(mu_inf, K_ALLOW, h, L)
            bv_fixed_h = bound_value(mu_inf, K_ALLOW, h_fixed, L)
            rows.append(dict(
                config=c["config"], arena=c["arena"], scenario=c["scenario"],
                phi=phi, snr=snr, delta_sigma_ref=magnitude,
                K=round(K, 4), F=round(F, 4), mu_inf=round(mu_inf, 4), k=K_ALLOW,
                h=round(h, 4), L=L, bound_value=bv, bound_vacuous=bool(bv >= 1.0),
                bound_below_resolution_floor=bool(bv < RESOLUTION_FLOOR),
                # mu_inf-only comparison: bound at the phi=0.95 arena's OWN
                # threshold, held fixed across phi, isolating the effect of
                # mu_inf shrinking from the confound of h growing near the
                # unit root (h itself is a function of phi via calibration).
                h_fixed_ref=round(h_fixed, 4), bound_at_fixed_h=bv_fixed_h,
                bound_vacuous_fixed_h=bool(bv_fixed_h >= 1.0),
                observed_detect_est=c["observed_detect_est"],
                observed_detect_known=obs_known,
            ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    n_3s = df[df.delta_sigma_ref == 3.0]
    n_3s_vacuous = int(n_3s.bound_vacuous.sum())
    n_3s_total = len(n_3s)
    n_3s_idle = int(((n_3s.bound_value >= 1.0) | (n_3s.bound_value < RESOLUTION_FLOOR)).sum())
    print(f"3-sigma cells: {n_3s_vacuous}/{n_3s_total} bound_vacuous (>=1); "
          f"{n_3s_idle}/{n_3s_total} EMPIRICALLY IDLE (vacuous OR below the "
          f"1/{N_REPS_RESOLUTION} resolution floor -- no cell is simultaneously "
          f"checkable and non-vacuous at this grid's replication budget)")
    print(df[df.delta_sigma_ref == 3.0]
          .sort_values(["config", "phi", "snr", "L"])
          [["config", "phi", "snr", "h", "mu_inf", "bound_value", "bound_vacuous",
            "bound_below_resolution_floor", "h_fixed_ref", "bound_at_fixed_h",
            "bound_vacuous_fixed_h"]]
          .to_string(index=False))
    print(f"\nwrote {OUT_PATH}")
    return df


if __name__ == "__main__":
    main()
