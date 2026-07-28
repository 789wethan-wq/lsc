"""exp52b + flagship detection-vs-horizon curve -- a single targeted
run at the paper's own body arena (SPEC_R8_missing_experiments.md S9,
scoped down; pre-registered in experiments/CHANGELOG.md 2026-07-27
BEFORE this script was run).

exp52 (CORRECTED) found the Proposition 1(b) finite-horizon bound
EMPIRICALLY IDLE at every published phi=0.95 cell -- vacuous (>=1) or
below the 1/500 resolution floor. That makes the detection-vs-horizon
SHAPE prediction (innovation CUSUM rises during the transient then
flattens; raw CUSUM keeps rising) the only remaining empirical
evidence for fast-or-never at the paper's actual body arena. This is
NOT the full exp45-gated exp51 -- one cell only (grid_v1's flagship:
ar1_snr0.5, phi=0.95, level_3s), reconstructed through grid_v1's exact
seeds so it is directly comparable to the published 0.554 / 0.990
detect rates.

exp52b: splits observed lsc_kalman_cusum alarms into transient
(t < break_time + T*) vs post-transient (t >= break_time + T*), where
T* = ceil(log(0.05)/log(rho)), rho = phi(1-K) the steady-state
innovation-mean decay rate -- and compares the OBSERVED post-transient
alarm rate (among replicates that survived the transient without
alarming) to exp52's already-computed Proposition 2 bound for this
exact cell. This is the first direct observed-vs-bound comparison at
the per-replicate level Proposition 1(b) has had.

Outputs:
  paper_assets/exp52b_flagship_curve.csv    -- P(detect by h), both detectors
  paper_assets/exp52b_transient_split.csv   -- the bound-vs-observed comparison
  paper_assets/exp52b_flagship_perrep.csv   -- per-replicate long file (SPEC S0.1)
  paper_assets/exp52b_flagship_curve.png    -- the curve, both detectors on one axis

Usage: python experiments/exp52b_flagship_curve.py
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.diagnostics.alarms import calibrate
from lsc.eval.detectors import make_innovation_cusum_detector, make_raw_cusum_detector
from lsc.models import KalmanModel
from lsc.theory import riccati_steady_state

REPO_ROOT = Path(__file__).resolve().parent.parent
A = REPO_ROOT / "paper_assets"

OUT_CURVE = A / "exp52b_flagship_curve.csv"
OUT_SPLIT = A / "exp52b_transient_split.csv"
OUT_PERREP = A / "exp52b_flagship_perrep.csv"
OUT_PNG = A / "exp52b_flagship_curve.png"

PHI, Q, R, T, N_TRAIN = 0.95, 0.04875, 1.0, 500, 125
FAR = 0.05
SEED_CAL, SEED_EVAL = 100_000, 200_000
N_REPS = 500
BREAK_TIME = 250
H_GRID = list(range(10, 251, 10))


def published_rate(method: str) -> float:
    df = pd.read_csv(A / "grid_v1_results.csv")
    m = df[(df.arena == "ar1_snr0.5") & (df.scenario == "level_3s") & (df.method == method)]
    return float(m.detect_rate.iloc[0]) if len(m) else float("nan")


def main() -> None:
    t0 = time.time()
    null_dgp = AR1StateDGP(phi=PHI, q=Q, r=R)
    break_dgp = AR1StateDGP(phi=PHI, q=Q, r=R,
                            breaks=[BreakSpec(kind="level", time_frac=0.5, magnitude=3.0)])
    assert break_dgp.breaks[0].time(T) == BREAK_TIME

    detectors = {
        "lsc_kalman_cusum": make_innovation_cusum_detector(lambda: KalmanModel("ar1"), N_TRAIN),
        "raw_cusum": make_raw_cusum_detector(N_TRAIN),
    }
    calibrated = {name: calibrate(name, fn, null_dgp, T, n_reps=N_REPS, far=FAR, seed0=SEED_CAL)
                 for name, fn in detectors.items()}

    perrep_rows = []
    delays = {name: [] for name in detectors}      # None if no post-break alarm
    alarm_idx = {name: [] for name in detectors}
    for i in range(N_REPS):
        Y = break_dgp.sample(T, seed=SEED_EVAL + i).Y
        for name, det in calibrated.items():
            at = det.alarm_time(Y)
            alarm_idx[name].append(at)
            delay = (at - BREAK_TIME) if (at is not None and at >= BREAK_TIME) else None
            delays[name].append(delay)
            score = det.score_fn(Y)
            finite = score[np.isfinite(score)]
            perrep_rows.append(dict(
                rep_id=i, arena="ar1_snr0.5", scenario="level_3s", channel="level",
                vol_mult=np.nan, snr=0.5, phi=PHI, detector=name,
                detected=int(delay is not None), alarm_index=at,
                score_max=float(finite.max()) if len(finite) else float("-inf"),
                threshold=det.threshold, seed=SEED_EVAL + i,
            ))

    detect_rate = {name: float(np.mean([d is not None for d in delays[name]]))
                  for name in detectors}
    pub = {name: published_rate(name) for name in detectors}
    reproduced = {name: bool(np.isclose(detect_rate[name], pub[name], atol=1e-9))
                 for name in detectors}
    print(f"[{time.time()-t0:5.0f}s] reproduction check: " +
          ", ".join(f"{n}={detect_rate[n]:.3f} (published {pub[n]:.3f}, "
                    f"match={reproduced[n]})" for n in detectors))

    curve_rows = []
    for h in H_GRID:
        for name in detectors:
            p_h = float(np.mean([d is not None and d <= h for d in delays[name]]))
            curve_rows.append(dict(detector=name, h=h, p_detect_by_h=p_h))
    df_curve = pd.DataFrame(curve_rows)
    df_curve.to_csv(OUT_CURVE, index=False)

    def gain(name, h_lo, h_hi):
        row = lambda h: df_curve[(df_curve.detector == name) & (df_curve.h == h)].p_detect_by_h.iloc[0]
        return row(h_hi) - row(h_lo)

    gain_kalman = gain("lsc_kalman_cusum", 60, 250)
    gain_raw = gain("raw_cusum", 60, 250)
    print(f"[{time.time()-t0:5.0f}s] gain(60->250): lsc_kalman_cusum={gain_kalman:+.4f} "
          f"(H_flagship predicts <0.05) raw_cusum={gain_raw:+.4f} (predicts >0.15)")

    # exp52b: transient / post-transient split for lsc_kalman_cusum
    _, K, F = riccati_steady_state(PHI, Q, R)
    rho = PHI * (1.0 - K)
    T_star = int(np.ceil(np.log(0.05) / np.log(rho)))
    cutoff = BREAK_TIME + T_star

    kalman_delays = delays["lsc_kalman_cusum"]
    kalman_alarm_idx = alarm_idx["lsc_kalman_cusum"]
    survived_transient = [at is None or at >= cutoff for at in kalman_alarm_idx]
    n_survived = int(np.sum(survived_transient))
    post_transient_alarms = [
        (at is not None and cutoff <= at <= BREAK_TIME + 250)
        for at, survived in zip(kalman_alarm_idx, survived_transient) if survived
    ]
    observed_post_transient_rate = float(np.mean(post_transient_alarms)) if n_survived else float("nan")

    from lsc.theory import mu_infinity
    sigma_ref = float(np.sqrt(Q / (1 - PHI**2)))
    delta = 3.0 * sigma_ref
    mu_inf = mu_infinity(delta, PHI, K, F)
    h_thresh = calibrated["lsc_kalman_cusum"].threshold
    L_post_transient = 250 - T_star
    bound_at_split = float((L_post_transient + 1) * np.exp(-2.0 * (0.5 - mu_inf) * h_thresh))

    split_row = dict(
        phi=PHI, snr=0.5, rho=round(rho, 4), T_star=T_star, cutoff_time=cutoff,
        n_survived_transient=n_survived, n_total=N_REPS,
        observed_post_transient_rate=observed_post_transient_rate,
        mu_inf=round(mu_inf, 4), k=0.5, h=round(h_thresh, 4), L=L_post_transient,
        bound_value=bound_at_split, bound_vacuous=bool(bound_at_split >= 1.0),
        observed_leq_bound=(bool(observed_post_transient_rate <= bound_at_split)
                            if bound_at_split < 1.0 else None),
    )
    pd.DataFrame([split_row]).to_csv(OUT_SPLIT, index=False)
    print(f"[{time.time()-t0:5.0f}s] exp52b: rho={rho:.4f} T*={T_star} "
          f"observed_post_transient_rate={observed_post_transient_rate:.4f} "
          f"(n_survived={n_survived}/{N_REPS}) vs bound_value={bound_at_split:.4g} "
          f"(vacuous={split_row['bound_vacuous']})")

    pd.DataFrame(perrep_rows).to_csv(OUT_PERREP, index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, marker in (("lsc_kalman_cusum", "o-"), ("raw_cusum", "s--")):
        sub = df_curve[df_curve.detector == name]
        ax.plot(sub.h, sub.p_detect_by_h, marker, ms=3, label=name)
    ax.axvline(T_star, color="gray", ls=":", label=f"transient end T*={T_star}")
    ax.set_xlabel("h (observations post-break)")
    ax.set_ylabel("P(detect by h)")
    ax.set_title("Detection-vs-horizon: flagship cell (phi=0.95, SNR=0.5, 3-sigma level shift)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)

    print(f"\n[{time.time()-t0:5.0f}s] wrote {OUT_CURVE}\nwrote {OUT_SPLIT}\n"
          f"wrote {OUT_PERREP}\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
