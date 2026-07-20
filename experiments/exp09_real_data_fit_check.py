"""exp09 -- model-fit diagnostic for the real-data application (SS9).

The real-data pipeline (real_data.py) fits an AR(1)+observation-noise
Kalman filter (KalmanModel("ar1")) on each rolling training window and
proceeds straight to detector calibration / alarm attribution, without
checking whether AR(1)+noise is an adequate description of the series
in that window. This script adds that check, reusing the exact segment
loop and fit call from real_data.run() (same NT/NM, same fitted_null
logic) -- it does not refit anything differently, just extracts and
diagnoses the standardized one-step innovations that were already an
implicit part of the pipeline.

For each series x rolling training window:
  1. Fit KalmanModel("ar1") on Y[:NT] (identical to real_data.fitted_null
     / composite detector calibration), filter over Y[:NT], take the
     standardized one-step innovations e_t for that training window.
  2. Ljung-Box test (statsmodels acorr_ljungbox) on e_t at two lag
     choices: one annual-cycle (12 for monthly series, 4 for quarterly
     GDP) and one two-year (24 monthly / 8 quarterly), per window.
  3. Model-implied ACF at lags 1-5 (rho^k for AR(1) with fitted phi,
     translated through the observation-noise dilution: for Y_t =
     S_t + v_t with S_t AR(1)(phi), ACF_Y(k) = phi^k / (1 + r/P_stat),
     P_stat = q/(1-phi^2)) vs the training window's raw sample ACF of Y
     at lags 1-5; report max |difference|.

Does NOT touch the paper's attribution claims. Report only.

Usage: python experiments/exp09_real_data_fit_check.py
Output: paper_assets/exp09_ljungbox_table.csv (per-window rows),
        printed per-series summary.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

sys.path.insert(0, str(Path(__file__).parent))
from real_data import SERIES, load_series  # noqa: E402
from lsc.models import KalmanModel  # noqa: E402

LAGS_MONTHLY = (12, 24)
LAGS_QUARTERLY = (4, 8)
ALPHA = 0.05


def sample_acf(y: np.ndarray, max_lag: int) -> np.ndarray:
    y = y - y.mean()
    c0 = np.dot(y, y) / len(y)
    return np.array([np.dot(y[k:], y[:-k]) / len(y) / c0
                      for k in range(1, max_lag + 1)])


def model_acf(phi: float, q: float, r: float, max_lag: int) -> np.ndarray:
    p_stat = q / (1.0 - phi**2)
    denom = p_stat + r
    return np.array([phi**k * p_stat / denom for k in range(1, max_lag + 1)])


def run_series(key: str) -> pd.DataFrame:
    cfg = SERIES[key]
    g = load_series(cfg, live=False)
    NT, NM = cfg["n_train"], cfg["n_monitor"]
    lags = LAGS_QUARTERLY if key == "gdp" else LAGS_MONTHLY
    rows = []
    seg_id = 0
    for start in range(0, len(g) - (NT + NM) + 1, NM):
        seg = g.iloc[start:start + NT + NM]
        Y_train = seg.values[:NT]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = KalmanModel("ar1").fit(Y_train)
            est = m.filter(Y_train)
        e = est.innovations
        e = e[np.isfinite(e)]
        p = est.params
        phi = float(np.clip(p["ar.L1"], 0.01, 0.99))
        q = max(float(p["sigma2.ar"]), 1e-8)
        r = max(float(p["sigma2.irregular"]), 1e-8)

        lb = acorr_ljungbox(e, lags=list(lags), return_df=True)
        row = dict(series=key, segment=seg_id,
                   window_start=f"{seg.index[0]:%Y-%m}",
                   window_end=f"{seg.index[NT - 1]:%Y-%m}",
                   n_train=NT, phi=round(phi, 3))
        for lag in lags:
            row[f"lb_pvalue_lag{lag}"] = round(
                float(lb.loc[lag, "lb_pvalue"]), 4)

        sample = sample_acf(Y_train, 5)
        model = model_acf(phi, q, r, 5)
        row["acf_max_abs_diff_lag1_5"] = round(
            float(np.max(np.abs(sample - model))), 4)
        rows.append(row)
        seg_id += 1
    return pd.DataFrame(rows)


def main() -> None:
    all_rows = []
    for key in ("indpro", "gdp", "gs10", "unrate"):
        df = run_series(key)
        all_rows.append(df)
        lags = LAGS_QUARTERLY if key == "gdp" else LAGS_MONTHLY
        print(f"\n=== {key} ({SERIES[key]['fred_id']}): "
              f"{len(df)} windows ===")
        for lag in lags:
            col = f"lb_pvalue_lag{lag}"
            frac_pass = float((df[col] > ALPHA).mean())
            print(f"  Ljung-Box lag={lag:2d}: p-value range "
                  f"[{df[col].min():.4f}, {df[col].max():.4f}], "
                  f"median {df[col].median():.4f}, "
                  f"{frac_pass:.0%} of windows pass (p>{ALPHA})")
        print(f"  ACF max|sample-model| lags 1-5: range "
              f"[{df['acf_max_abs_diff_lag1_5'].min():.4f}, "
              f"{df['acf_max_abs_diff_lag1_5'].max():.4f}], "
              f"median {df['acf_max_abs_diff_lag1_5'].median():.4f}")

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv("paper_assets/exp09_ljungbox_table.csv", index=False)
    print("\nwrote paper_assets/exp09_ljungbox_table.csv")


if __name__ == "__main__":
    main()
