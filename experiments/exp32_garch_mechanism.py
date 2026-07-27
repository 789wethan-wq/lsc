"""exp32 -- does GARCH's conditional-variance path track the true break
regime, or is it flat/uninformative? (SPEC R4 M1, pre-registered in
experiments/CHANGELOG.md 2026-07-25 BEFORE this script was run.)

Distinguishes two explanations for the GARCH(1,1) floor-only-at-subtle-
breaks result (Related Work, GARCH paragraph): "GARCH structurally
can't represent this DGP's break" vs. "GARCH represents it fine but the
CUSUM-on-standardized-residuals wrapper is underpowered at that effect
size." For each of exp15's 12 cells, pools every post-training-prefix
time point across n_reps replicates into (score_t, true_regime_label_t)
pairs -- true_regime_label_t = 1{t >= break_time} -- and computes:

  - GARCH's fitted conditional-variance path sigma2_t
    (garch_detector.garch_conditional_variance_path)
  - raw rung's squared standardized-residual path z_t^2 (frozen
    training-prefix moments, same convention as raw_var_cusum_score)
  - ARIMA rung's squared standardized-residual path z_t^2
    (arima_standardized_residuals, same convention as
    arima_var_cusum_score)

against the true regime, via Spearman correlation and AUC (Mann-Whitney
U, rank-based -- scale-invariant, so GARCH's sigma2 and raw/ARIMA's z^2
being on different natural scales does not bias the comparison).

Same 2x2x3 grid as exp15 (channel in {r, q} x vol_mult in {1.5, 3.0} x
SNR in {0.1, 0.5, 2.0}), phi=0.95, T=500, n_train=125, n_reps=500. NEW
disjoint seed block (400000+, evaluation-only -- no calibration/
threshold needed for this diagnostic, so no calibration seed is drawn)
-- unlike exp30, this is a standalone diagnostic with no downstream
paired-decomposition requirement, so a fresh block does not sacrifice
anything.

Usage: python experiments/exp32_garch_mechanism.py [n_reps]
Output: paper_assets/exp32_garch_mechanism.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from garch_detector import garch_conditional_variance_path
from lsc.benchmarks.arima import arima_standardized_residuals
from lsc.benchmarks.variance import training_moments
from lsc.dgp import AR1StateDGP, BreakSpec

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "paper_assets" / "exp32_garch_mechanism.csv"

PHI, R, T, N_TRAIN = 0.95, 1.0, 500, 125
SEED_EVAL = 400_000  # fresh, evaluation-only block (no calibration needed)

CHANNELS = ("r", "q")
VOL_MULTS = (1.5, 3.0)
SNRS = (0.1, 0.5, 2.0)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney-U AUC: P(score | label=1) > P(score | label=0),
    ties counted as 0.5. Rank-based, scale-invariant."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    sum_ranks_pos = ranks[labels.astype(bool)].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def run_cell(snr: float, channel: str, vol_mult: float, n_reps: int) -> dict:
    q = snr * (1 - PHI**2) * R
    kind = "variance" if channel == "r" else "state_var"
    break_dgp = AR1StateDGP(phi=PHI, q=q, r=R,
                            breaks=[BreakSpec(kind=kind, time_frac=0.5, vol_mult=vol_mult)])
    break_time = break_dgp.breaks[0].time(T)

    garch_scores, raw_scores, arima_scores, labels = [], [], [], []
    for i in range(n_reps):
        Y = break_dgp.sample(T, seed=SEED_EVAL + i).Y

        sigma2 = garch_conditional_variance_path(Y, N_TRAIN)[N_TRAIN:]
        mu, sd = training_moments(Y, N_TRAIN)
        raw_z2 = (((Y - mu) / sd) ** 2)[N_TRAIN:]
        arima_z2 = (arima_standardized_residuals(Y, N_TRAIN) ** 2)[N_TRAIN:]
        lbl = (np.arange(N_TRAIN, T) >= break_time).astype(float)

        garch_scores.append(sigma2)
        raw_scores.append(raw_z2)
        arima_scores.append(arima_z2)
        labels.append(lbl)

    garch_scores = np.concatenate(garch_scores)
    raw_scores = np.concatenate(raw_scores)
    arima_scores = np.concatenate(arima_scores)
    labels = np.concatenate(labels)
    n_points = len(labels)

    rho_garch, _ = spearmanr(garch_scores, labels)
    rho_raw, _ = spearmanr(raw_scores, labels)
    rho_arima, _ = spearmanr(arima_scores, labels)

    return dict(
        channel=channel, snr=snr, vol_mult=vol_mult, n_reps=n_reps, n_points=n_points,
        spearman_garch=float(rho_garch), auc_garch=auc(garch_scores, labels),
        spearman_raw=float(rho_raw), auc_raw=auc(raw_scores, labels),
        spearman_arima=float(rho_arima), auc_arima=auc(arima_scores, labels),
    )


def _load_existing() -> pd.DataFrame:
    if OUT_PATH.exists():
        return pd.read_csv(OUT_PATH)
    return pd.DataFrame(columns=["channel", "snr", "vol_mult", "n_reps"])


def _already_done(existing, channel, snr, vol_mult, n_reps):
    if existing.empty:
        return None
    m = existing[(existing.channel == channel) & np.isclose(existing.snr, snr)
                & np.isclose(existing.vol_mult, vol_mult) & (existing.n_reps == n_reps)]
    return m.iloc[0].to_dict() if len(m) else None


if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    t0 = time.time()
    existing = _load_existing()
    rows = []
    for channel in CHANNELS:
        for vol_mult in VOL_MULTS:
            for snr in SNRS:
                cached = _already_done(existing, channel, snr, vol_mult, n_reps)
                if cached is not None:
                    rows.append(cached)
                    print(f"[{time.time()-t0:6.0f}s] channel={channel} SNR={snr} "
                          f"vol_mult={vol_mult}: reused", flush=True)
                    continue
                out = run_cell(snr, channel, vol_mult, n_reps)
                rows.append(out)
                print(f"[{time.time()-t0:6.0f}s] channel={channel} SNR={snr} "
                      f"vol_mult={vol_mult}: "
                      f"GARCH rho={out['spearman_garch']:.3f} AUC={out['auc_garch']:.3f} | "
                      f"raw rho={out['spearman_raw']:.3f} AUC={out['auc_raw']:.3f} | "
                      f"ARIMA rho={out['spearman_arima']:.3f} AUC={out['auc_arima']:.3f}",
                      flush=True)
    df = pd.DataFrame(rows).sort_values(["channel", "vol_mult", "snr"]).reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote {OUT_PATH}")
