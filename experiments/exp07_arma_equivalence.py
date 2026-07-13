"""M1 [GATE] — ARMA(1,1) equivalence of the ARIMA and Kalman rungs.

For S_t = phi S_{t-1} + w_t (var q), Y_t = S_t + v_t (var r), the
observable Y has an exact ARMA(1,1) reduced form (lsc.theory
.arma11_representation): differencing by (1 - phi L) leaves an MA(1),
whose invertible root gives MA parameter theta and innovation variance
sigma_eps^2, with the machine-precision identities

    sigma_eps^2 == F  (the Riccati innovation variance, Proposition 1)
    theta       == rho == phi (1 - K)  (the innovation-mean decay rate).

So at the steady state with correct parameters the Kalman one-step
innovations and the ARMA(1,1) innovations are the SAME linear
innovations of the SAME Gaussian process — the ARIMA and Kalman rungs
of the whitening ladder are one filter, not two. This script confirms
that numerically on null paths, both with estimated parameters (the
ladder's actual operating condition) and with true parameters (which
isolates estimation error from any structural difference).

Decision rule (pre-registered, CHANGELOG M0 2026-07-13):
  A1  rho_bar >= 0.95  -> equivalent; two-rung reframing.
  A2  rho_bar <  0.95  -> STOP; estimator bug, diagnose before M2-M4.

Deliverables: paper_assets/arma_equivalence.csv (per-SNR/arm summary),
paper_assets/arma_equivalence_orders.csv (ARIMA order distribution),
paper_assets/arma_equivalence.png (series overlay + rho-by-SNR).

Usage: python experiments/exp07_arma_equivalence.py [n_paths]
"""
from __future__ import annotations

import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from lsc.benchmarks.arima import arima_standardized_residuals, fit_arima_prefix
from lsc.dgp import AR1StateDGP
from lsc.models import KalmanModel
from lsc.theory import arma11_representation, steady_state_innovations

A = "paper_assets"

# phi=0.95, r=1.0 fixed; q sets the SNR = q/(r(1-phi^2)) marginal-state
# ratio, matching grid_v4_varbench_core arenas exactly.
ARENAS = {"0.1": 0.00975, "0.5": 0.04875, "2.0": 0.195}
PHI, R = 0.95, 1.0
T = 500
TRAIN_FRAC = 0.25
SEED0 = 100000  # calibration (null) block


def kalman_innovations_estimated(Y: np.ndarray, n_train: int) -> np.ndarray:
    """Standardized innovations from the frozen training-prefix Kalman
    (ar1) model — the latent rung's actual operating quantity."""
    m = KalmanModel("ar1").fit(Y[:n_train])
    return m.filter(Y).innovations


def arima_innovations_fixed(Y: np.ndarray, n_train: int,
                            order: tuple = (1, 0, 1)) -> np.ndarray:
    """Standardized one-step residuals with the ARIMA order FORCED (no
    AIC selection), parameters fitted on the training prefix and frozen.
    Isolates the AIC order-selection wedge from estimation error."""
    from statsmodels.tsa.arima.model import ARIMA

    Y = np.asarray(Y, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(Y[:n_train], order=order).fit()
        full = ARIMA(Y, order=order).filter(res.params)
        return np.asarray(full.standardized_forecasts_error).ravel()


def arima_innovations_true(Y: np.ndarray, phi: float, q: float, r: float
                           ) -> np.ndarray:
    """Standardized one-step residuals of the ARMA(1,1) with the TRUE
    reduced-form parameters (phi, -theta, sigma_eps^2), via the
    statsmodels ARMA filter — a code path entirely independent of the
    hand-written steady-state Kalman recursion it is compared against."""
    from statsmodels.tsa.arima.model import ARIMA

    theta, sigma2 = arma11_representation(phi, q, r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(np.asarray(Y, float), order=(1, 0, 1), trend="n").filter(
            [phi, -theta, sigma2])
        return np.asarray(res.standardized_forecasts_error).ravel()


def _pair_stats(a: np.ndarray, b: np.ndarray, m0: int) -> tuple[float, float]:
    """(Pearson rho, max|delta|) over the monitoring region t >= m0."""
    x, y = a[m0:], b[m0:]
    rho = float(np.corrcoef(x, y)[0, 1])
    return rho, float(np.max(np.abs(x - y)))


def run(n_paths: int) -> None:
    n_train = int(round(TRAIN_FRAC * T))
    m0 = n_train  # post-burn-in monitoring region
    rows, order_rows = [], []
    # per-(snr) accumulators for the rho-by-t curve (estimated arm)
    by_t = {snr: [] for snr in ARENAS}
    overlay = {}  # snr -> (t, e_kal, e_arima) for one representative path

    for snr, q in ARENAS.items():
        dgp = AR1StateDGP(phi=PHI, q=q, r=R)  # null (no breaks)
        orders = Counter()
        est_rho, est_md, tru_rho, tru_md = [], [], [], []
        f101_rho, f101_md = [], []
        e_est_stack_kal, e_est_stack_ari = [], []
        for i in range(n_paths):
            Y = dgp.sample(T, seed=SEED0 + i).Y
            # estimated arm (ladder's real operating condition: AIC order)
            e_kal = kalman_innovations_estimated(Y, n_train)
            e_ari = arima_standardized_residuals(Y, n_train)
            order, _ = fit_arima_prefix(Y, n_train)
            orders[order] += 1
            r_est, d_est = _pair_stats(e_kal, e_ari, m0)
            est_rho.append(r_est); est_md.append(d_est)
            e_est_stack_kal.append(e_kal); e_est_stack_ari.append(e_ari)
            # forced-(1,0,1) arm: isolates the AIC-selection wedge
            e_101 = arima_innovations_fixed(Y, n_train, (1, 0, 1))
            r_101, d_101 = _pair_stats(e_kal, e_101, m0)
            f101_rho.append(r_101); f101_md.append(d_101)
            # true-parameter arm (isolates estimation error)
            e_kal_t = steady_state_innovations(Y, PHI, q, R)
            e_ari_t = arima_innovations_true(Y, PHI, q, R)
            r_tru, d_tru = _pair_stats(e_kal_t, e_ari_t, m0)
            tru_rho.append(r_tru); tru_md.append(d_tru)
            if i == 0:
                t = np.arange(m0, T)
                overlay[snr] = (t, e_kal[m0:], e_ari[m0:])

        # rho-by-t (cross-path correlation at each t): exposes startup
        K = np.vstack(e_est_stack_kal); Aa = np.vstack(e_est_stack_ari)
        for t in range(m0, T):
            xt, yt = K[:, t], Aa[:, t]
            if np.std(xt) > 0 and np.std(yt) > 0:
                by_t[snr].append((t, float(np.corrcoef(xt, yt)[0, 1])))

        for arm, rr, dd in [("estimated", est_rho, est_md),
                            ("estimated_forced101", f101_rho, f101_md),
                            ("true_params", tru_rho, tru_md)]:
            rr = np.asarray(rr); dd = np.asarray(dd)
            rows.append(dict(
                snr=snr, arm=arm, n_paths=n_paths,
                rho_median=float(np.median(rr)),
                rho_iqr_lo=float(np.percentile(rr, 25)),
                rho_iqr_hi=float(np.percentile(rr, 75)),
                rho_min=float(np.min(rr)),
                max_abs_delta_median=float(np.median(dd)),
                max_abs_delta_max=float(np.max(dd)),
            ))
        for order, cnt in sorted(orders.items()):
            order_rows.append(dict(snr=snr, order=str(order), count=cnt,
                                   frac=cnt / n_paths))
        rb = np.median([r for _, r in by_t[snr]])
        print(f"SNR {snr}: est(AIC) rho_med={np.median(est_rho):.4f} "
              f"min={np.min(est_rho):.4f} | est(1,0,1) rho_med={np.median(f101_rho):.4f} "
              f"min={np.min(f101_rho):.4f} | true rho_med={np.median(tru_rho):.6f} "
              f"maxdelta={np.max(tru_md):.2e} | order-mode={orders.most_common(1)[0]} "
              f"| rho-by-t median={rb:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{A}/arma_equivalence.csv", index=False)
    od = pd.DataFrame(order_rows)
    od.to_csv(f"{A}/arma_equivalence_orders.csv", index=False)

    rho_bar = df[df.arm == "estimated"].rho_median.median()
    rho_bar_true = df[df.arm == "true_params"].rho_median.median()
    branch = "A1 (equivalent)" if rho_bar >= 0.95 else "A2 (STOP — bug)"
    print(f"\nrho_bar (estimated, median over SNR) = {rho_bar:.4f}")
    print(f"rho_bar (true params)                = {rho_bar_true:.6f}")
    print(f"GATE: {branch}")

    _figure(df, od, by_t, overlay)
    print(f"wrote {A}/arma_equivalence.csv, _orders.csv, .png")


def _figure(df, od, by_t, overlay) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    # (a) series overlay, one null path, SNR 0.5
    t, ek, ea = overlay["0.5"]
    axes[0].plot(t, ek, lw=0.8, label="Kalman innovation", color="C0")
    axes[0].plot(t, ea, lw=0.8, ls="--", label="ARIMA residual", color="C1")
    axes[0].set_title("(a) standardized series, one null path (SNR 0.5)",
                      fontsize=9)
    axes[0].set_xlabel("t"); axes[0].legend(fontsize=7)
    # (b) rho-by-t (estimated arm), each SNR — exposes startup transient
    for snr in ARENAS:
        arr = np.asarray(by_t[snr])
        axes[1].plot(arr[:, 0], arr[:, 1], lw=0.9, label=f"SNR {snr}")
    axes[1].axhline(0.95, color="k", ls=":", lw=0.8)
    axes[1].set_title("(b) cross-path corr by t (estimated)", fontsize=9)
    axes[1].set_xlabel("t"); axes[1].set_ylabel(r"$\rho_t$")
    axes[1].set_ylim(0.8, 1.001); axes[1].legend(fontsize=7)
    # (c) rho median (estimated) with IQR whiskers, by SNR
    est = df[df.arm == "estimated"].set_index("snr").loc[list(ARENAS)]
    x = np.arange(len(ARENAS))
    axes[2].errorbar(x, est.rho_median,
                     yerr=[est.rho_median - est.rho_iqr_lo,
                           est.rho_iqr_hi - est.rho_median],
                     fmt="o", capsize=4, color="C0", label="median (IQR)")
    axes[2].plot(x, est.rho_min, "v", color="C3", label="min")
    axes[2].axhline(0.95, color="k", ls=":", lw=0.8, label="A1 threshold")
    axes[2].set_xticks(x); axes[2].set_xticklabels(list(ARENAS))
    axes[2].set_title(r"(c) $\rho$ by SNR (estimated)", fontsize=9)
    axes[2].set_xlabel("SNR"); axes[2].set_ylim(0.85, 1.005)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{A}/arma_equivalence.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
