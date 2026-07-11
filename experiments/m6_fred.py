"""M6 — real-data application: US industrial production (FRED INDPRO).

ILLUSTRATIVE ONLY (SPEC §4.5 / §8): real data has no ground-truth break
dates. NBER business-cycle peak months are used as documented external
reference events; alarms are compared to them without any claim that
they are "the truth".

Design (fully causal):
  - Series: monthly log-growth of INDPRO (%), 1948 onward.
  - Rolling monitoring: fit model params on a 120-month training
    window, monitor the following 60 months, roll forward 60 months.
  - Threshold calibration per segment by PARAMETRIC BOOTSTRAP: fit the
    AR(1) SSM on the training window, simulate n_cal null series from
    the fitted parameters, calibrate each detector to 5% FAR per
    monitoring window — the same calibration machinery as the
    simulation study, with the fitted model as the null.
  - Detectors: LSC composite, LSC innovation CUSUM, raw-Y CUSUM.

Outputs: paper_assets/m6_fred_alarms.csv, m6_fred_figure.png,
m6_fred_summary.csv.

Usage: python experiments/m6_fred.py [n_cal]
"""
from __future__ import annotations

import io
import sys
import time
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lsc.dgp import AR1StateDGP
from lsc.diagnostics.alarms import calibrate
from lsc.eval.detectors import (
    make_composite_detector,
    make_innovation_cusum_detector,
    make_raw_cusum_detector,
)
from lsc.models import KalmanModel

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=INDPRO"
START = "1948-01-01"
N_TRAIN, N_MONITOR = 120, 60
FAR_TARGET = 0.05

# NBER business-cycle peaks (recession starts), nber.org/cycles — used
# as reference events only.
NBER_PEAKS = [
    "1948-11", "1953-07", "1957-08", "1960-04", "1969-12", "1973-11",
    "1980-01", "1981-07", "1990-07", "2001-03", "2007-12", "2020-02",
]
NBER_TROUGHS = [
    "1949-10", "1954-05", "1958-04", "1961-02", "1970-11", "1975-03",
    "1980-07", "1982-11", "1991-03", "2001-11", "2009-06", "2020-04",
]


def load_indpro() -> pd.Series:
    with urllib.request.urlopen(FRED_URL, timeout=60) as r:
        raw = r.read().decode()
    df = pd.read_csv(io.StringIO(raw), parse_dates=["observation_date"])
    df = df.set_index("observation_date")["INDPRO"].astype(float)
    growth = 100.0 * np.log(df).diff().dropna()
    return growth[growth.index >= START]


def fitted_null(Y_train: np.ndarray) -> AR1StateDGP:
    m = KalmanModel("ar1").fit(Y_train)
    p = dict(zip(m._param_names, m._params))
    phi = float(np.clip(p["ar.L1"], 0.01, 0.99))
    return AR1StateDGP(phi=phi, q=max(float(p["sigma2.ar"]), 1e-8),
                       r=max(float(p["sigma2.irregular"]), 1e-8))


def main(n_cal: int = 200) -> None:
    t0 = time.time()
    g = load_indpro()
    # de-mean by training window inside detectors; keep raw growth here
    print(f"loaded INDPRO growth: {len(g)} months "
          f"({g.index[0]:%Y-%m} .. {g.index[-1]:%Y-%m})")

    T_seg = N_TRAIN + N_MONITOR
    alarms = []
    seg_id = 0
    for start in range(0, len(g) - T_seg + 1, N_MONITOR):
        seg = g.iloc[start:start + T_seg]
        Y = seg.values
        null = fitted_null(Y[:N_TRAIN])
        detectors = {
            "lsc_composite": make_composite_detector(
                lambda: KalmanModel("ar1"), null, T_seg, N_TRAIN,
                n_scale_reps=50),
            "lsc_kalman_cusum": make_innovation_cusum_detector(
                lambda: KalmanModel("ar1"), N_TRAIN),
            "raw_cusum": make_raw_cusum_detector(N_TRAIN),
        }
        for name, fn in detectors.items():
            det = calibrate(name, fn, null, T_seg, n_reps=n_cal,
                            far=FAR_TARGET, seed0=100_000 + 1000 * seg_id)
            at = det.alarm_time(Y)
            if at is not None:
                alarms.append(dict(segment=seg_id, method=name,
                                   date=seg.index[at]))
        print(f"[{time.time()-t0:5.0f}s] segment {seg_id} "
              f"({seg.index[N_TRAIN]:%Y-%m} monitored)", flush=True)
        seg_id += 1

    adf = pd.DataFrame(alarms)
    adf.to_csv("paper_assets/m6_fred_alarms.csv", index=False)

    # summary: for each NBER peak, delay (months) to first alarm per method
    peaks = pd.to_datetime([p + "-01" for p in NBER_PEAKS])
    rows = []
    for method in adf.method.unique():
        dates = pd.DatetimeIndex(sorted(adf[adf.method == method].date))
        for peak in peaks:
            after = dates[(dates >= peak) & (dates <= peak + pd.DateOffset(months=24))]
            rows.append(dict(method=method, nber_peak=f"{peak:%Y-%m}",
                             first_alarm=(f"{after[0]:%Y-%m}" if len(after) else "none"),
                             delay_months=(int((after[0].year - peak.year) * 12
                                               + after[0].month - peak.month)
                                           if len(after) else np.nan)))
    summ = pd.DataFrame(rows)
    summ.to_csv("paper_assets/m6_fred_summary.csv", index=False)
    print(summ.pivot_table(index="nber_peak", columns="method",
                           values="delay_months").to_string())

    # figure
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(g.index, g.values, lw=0.5, color="gray", label="INDPRO growth (%)")
    for pk, tr in zip(NBER_PEAKS, NBER_TROUGHS):
        ax.axvspan(pd.Timestamp(pk + "-01"), pd.Timestamp(tr + "-01"),
                   color="red", alpha=0.15)
    colors = {"lsc_composite": "C0", "lsc_kalman_cusum": "C2", "raw_cusum": "C3"}
    offs = {"lsc_composite": 1.0, "lsc_kalman_cusum": 0.93, "raw_cusum": 0.86}
    for method, gr in adf.groupby("method"):
        for d in gr.date:
            ax.plot([d], [ax.get_ylim()[1] * offs[method]], marker="v",
                    color=colors[method], ms=6)
        ax.plot([], [], marker="v", ls="", color=colors[method], label=method)
    ax.legend(fontsize=8)
    ax.set_title("INDPRO monthly growth — causal alarms (5% FAR per 60-month "
                 "window, parametric-bootstrap calibration) vs NBER recessions "
                 "(shaded). Illustrative only.")
    fig.tight_layout()
    fig.savefig("paper_assets/m6_fred_figure.png", dpi=130)
    print("wrote paper_assets/m6_fred_*")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
