"""exp30 -- order-known ARIMA rung: isolate order-selection error from
coefficient (MLE) error in the q-channel Table 3 known-minus-estimated
gap (SPEC R3 M1, pre-registered in experiments/CHANGELOG.md 2026-07-25
BEFORE this script was run).

Three conditions on the identical q-channel cells Table 3/exp26 use:
  estimated     -- AIC picks the order, MLE fits coefficients (existing
                   arima_var_cusum rung, grid_v5_qbreak_results.csv)
  order_known   -- NEW: order FIXED at the true (1,0,1), MLE still fits
                   coefficients (order_known_var_cusum_score)
  known         -- true phi, q, r handed directly to the Kalman filter
                   (existing, exp26_known_param_variance.csv,
                   detect_known_kalman column)

gap_order_selection = detect(order_known) - detect(estimated)
gap_coefficient_noise = detect(known) - detect(order_known)
Sanity check: the two gaps should sum to the published
detect(known) - detect(estimated) (exp26's gap_kalman column) to
within rounding.

SEEDS (see CHANGELOG for why these are NOT a fresh disjoint block):
calibration=100000, evaluation=200000 -- the SAME standing blocks
exp26/grid_v5 use, so order_known is evaluated on the IDENTICAL
simulated paths as estimated and known; this is what makes the
decomposition and sanity check meaningful rather than three
independent Monte Carlo estimates. far_check=300000 (exp24's
convention) for the requested fresh-null FAR re-verification only.

Usage: python experiments/exp30_order_known_arima.py [n_reps]
Output: paper_assets/exp30_order_known_arima.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lsc.benchmarks.variance import order_known_var_cusum_score
from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.diagnostics.alarms import calibrate, empirical_far

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "paper_assets" / "exp30_order_known_arima.csv"

PHI, R, T, N_TRAIN = 0.95, 1.0, 500, 125
FAR = 0.05
SEED_CAL, SEED_EVAL, SEED_FAR_CHECK = 100_000, 200_000, 300_000

VOL_MULTS = (1.5, 3.0)
SNRS = (0.1, 0.5, 2.0)

TRUE_ORDER = (1, 0, 1)


def _published_gap_kalman(snr: float, vol_mult: float) -> tuple[float, float, float]:
    """(detect_est_arima, detect_known_kalman, gap_kalman) from exp26,
    channel=q, for the sanity check."""
    df = pd.read_csv(REPO_ROOT / "paper_assets" / "exp26_known_param_variance.csv")
    m = df[(df.channel == "q") & np.isclose(df.snr, snr) & np.isclose(df.vol_mult, vol_mult)]
    if not len(m):
        return float("nan"), float("nan"), float("nan")
    row = m.iloc[0]
    return float(row.detect_est_arima), float(row.detect_known_kalman), float(row.gap_kalman)


def run_cell(snr: float, vol_mult: float, n_reps: int) -> dict:
    q = snr * (1 - PHI**2) * R
    null_dgp = AR1StateDGP(phi=PHI, q=q, r=R)

    fn_order_known = lambda Y: order_known_var_cusum_score(Y, N_TRAIN, order=TRUE_ORDER)
    det_order_known = calibrate("order_known_var_cusum", fn_order_known, null_dgp, T,
                                n_reps=n_reps, far=FAR, seed0=SEED_CAL)

    break_dgp = AR1StateDGP(phi=PHI, q=q, r=R,
                            breaks=[BreakSpec(kind="state_var", time_frac=0.5, vol_mult=vol_mult)])
    break_time = break_dgp.breaks[0].time(T)
    paths = [break_dgp.sample(T, seed=SEED_EVAL + i).Y for i in range(n_reps)]
    detect_order_known = float(np.mean([
        det_order_known.alarm_time(Y) is not None and det_order_known.alarm_time(Y) >= break_time
        for Y in paths]))

    fresh_far = empirical_far(det_order_known, null_dgp, T, n_reps=n_reps, seed0=SEED_FAR_CHECK)

    detect_est_arima, detect_known_kalman, published_gap = _published_gap_kalman(snr, vol_mult)
    gap_order_selection = detect_order_known - detect_est_arima
    gap_coefficient_noise = detect_known_kalman - detect_order_known
    reconstructed_gap = gap_order_selection + gap_coefficient_noise
    sanity_ok = bool(np.isclose(reconstructed_gap, published_gap, atol=0.02))

    return dict(
        channel="q", snr=snr, vol_mult=vol_mult, n_reps=n_reps,
        threshold_order_known=det_order_known.threshold,
        empirical_far_calibration=float(
            (det_order_known.null_max_scores >= det_order_known.threshold).mean()),
        empirical_far_fresh_nulls=fresh_far,
        detect_est_arima=detect_est_arima,
        detect_order_known=detect_order_known,
        detect_known_kalman=detect_known_kalman,
        gap_order_selection=gap_order_selection,
        gap_coefficient_noise=gap_coefficient_noise,
        reconstructed_total_gap=reconstructed_gap,
        published_gap_kalman=published_gap,
        sanity_check_ok=sanity_ok,
    )


if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    t0 = time.time()
    rows = []
    for vol_mult in VOL_MULTS:
        for snr in SNRS:
            out = run_cell(snr, vol_mult, n_reps)
            rows.append(out)
            print(f"[{time.time()-t0:6.0f}s] SNR={snr} vol_mult={vol_mult}: "
                  f"order_known={out['detect_order_known']:.3f} "
                  f"(est={out['detect_est_arima']:.3f}, known={out['detect_known_kalman']:.3f}) "
                  f"gap_order_sel={out['gap_order_selection']:+.3f} "
                  f"gap_coef_noise={out['gap_coefficient_noise']:+.3f} "
                  f"recon_total={out['reconstructed_total_gap']:+.3f} "
                  f"published_total={out['published_gap_kalman']:+.3f} "
                  f"sanity_ok={out['sanity_check_ok']} "
                  f"fresh_FAR={out['empirical_far_fresh_nulls']:.3f}", flush=True)
    df = pd.DataFrame(rows).sort_values(["vol_mult", "snr"]).reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    all_ok = bool(df.sanity_check_ok.all())
    print(f"\nAll sanity checks pass: {all_ok}")
    if not all_ok:
        print("!!! at least one cell's gap decomposition does not reconstruct the "
              "published known-minus-estimated gap -- investigate before trusting "
              "the order-selection/coefficient-noise split below.")
    print(f"wrote {OUT_PATH}")
