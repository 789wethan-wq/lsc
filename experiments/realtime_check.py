"""m6x real-time vintage check (design registered in CHANGELOG m6x
entry): would the INDPRO alarms around the GFC and COVID have fired at
the same time on the data AS IT EXISTED each month (ALFRED vintages),
rather than on today's revised series?

Design (per episode): training window = the 120 months of log-growth
ending at the later-declared NBER peak; monitoring starts the month
after the peak. For each decision month m, download the vintage dated
m (data through m-1 due to publication lag), fit the null on the
vintage's training window, recalibrate every detector by parametric
bootstrap at 5% FAR (thresholds are vintage-specific because the
training data are), score the monitored months available in that
vintage, and record whether an alarm has occurred. The real-time alarm
month for a method is the first decision month with an alarm; we also
report the DATA month at which the score crossed. Robustness criterion
(registered): the revised-data alarm month must match within +-1 month.

Vintages are cached under data/vintages/. Usage:
  python experiments/realtime_check.py [n_cal]
Output: paper_assets/rd_realtime.csv
"""
from __future__ import annotations

import io
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from lsc.dgp import AR1StateDGP
from lsc.diagnostics.alarms import calibrate
from lsc.eval.detectors import (
    make_composite_detector,
    make_raw_cusum_detector,
    make_tail_cusum_detector,
)
from lsc.models import KalmanModel

ALFRED = ("https://alfred.stlouisfed.org/graph/alfredgraph.csv"
          "?id=INDPRO&vintage_date={d}")
N_TRAIN, N_MONITOR = 120, 60
FAR = 0.05

EPISODES = {
    "gfc": dict(peak="2007-12", decisions=pd.period_range(
        "2008-01", "2009-06", freq="M")),
    "covid": dict(peak="2020-02", decisions=pd.period_range(
        "2020-03", "2020-12", freq="M")),
}


def load_vintage(vintage_day: str) -> pd.Series:
    cache = Path(f"data/vintages/INDPRO_{vintage_day}.csv")
    if cache.exists():
        raw = cache.read_text()
    else:
        with urllib.request.urlopen(
                ALFRED.format(d=vintage_day), timeout=60) as r:
            raw = r.read().decode()
        cache.write_text(raw)
    df = pd.read_csv(io.StringIO(raw), parse_dates=["observation_date"])
    col = [c for c in df.columns if c.startswith("INDPRO")][0]
    s = df.set_index("observation_date")[col].replace(".", np.nan)
    s = s.astype(float).dropna()
    g = 100.0 * np.log(s).diff().dropna()
    return g[g.index >= "1948-01-01"]


def fitted_null(Y_train: np.ndarray) -> AR1StateDGP:
    m = KalmanModel("ar1").fit(Y_train)
    p = dict(zip(m._param_names, m._params))
    phi = float(np.clip(p["ar.L1"], 0.01, 0.99))
    return AR1StateDGP(phi=phi, q=max(float(p["sigma2.ar"]), 1e-8),
                       r=max(float(p["sigma2.irregular"]), 1e-8))


def episode_check(name: str, peak: str, decisions, n_cal: int,
                  t0: float) -> list[dict]:
    peak_ts = pd.Timestamp(peak + "-01")
    T_cal = N_TRAIN + N_MONITOR
    first_alarm: dict[str, dict] = {}
    for m in decisions:
        vday = f"{m.year}-{m.month:02d}-15"
        g = load_vintage(vday)
        train = g[g.index <= peak_ts].iloc[-N_TRAIN:]
        monitor = g[g.index > peak_ts]
        if len(train) < N_TRAIN or len(monitor) == 0:
            continue
        Y = np.concatenate([train.values, monitor.values])
        idx = train.index.append(monitor.index)
        # pad to the calibration length: detectors are strictly causal
        # (bit-identical no-lookahead tests), so trailing padding cannot
        # change scores at real time points; alarms are only accepted
        # at t < len(Y).
        Y_pad = np.concatenate([Y, np.zeros(T_cal - len(Y))]) \
            if len(Y) < T_cal else Y[:T_cal]
        null = fitted_null(train.values)
        detectors = {
            "lsc_composite": make_composite_detector(
                lambda: KalmanModel("ar1"), null, T_cal, N_TRAIN,
                n_scale_reps=50),
            "lsc_tail_cusum": make_tail_cusum_detector(
                lambda: KalmanModel("ar1"), N_TRAIN),
            "raw_cusum": make_raw_cusum_detector(N_TRAIN),
        }
        for mname, fn in detectors.items():
            if mname in first_alarm:
                continue
            det = calibrate(mname, fn, null, T_cal, n_reps=n_cal,
                            far=FAR, seed0=100_000)
            at = det.alarm_time(Y_pad)
            if at is not None and at < len(Y):
                first_alarm[mname] = dict(
                    episode=name, method=mname,
                    decision_month=str(m),
                    data_month=f"{idx[at]:%Y-%m}",
                    vintage=vday)
        print(f"[{time.time()-t0:5.0f}s] {name} decision {m}: "
              f"alarmed so far: {sorted(first_alarm)}", flush=True)
        if len(first_alarm) == 3:
            break
    for mname in ("lsc_composite", "lsc_tail_cusum", "raw_cusum"):
        if mname not in first_alarm:
            first_alarm[mname] = dict(episode=name, method=mname,
                                      decision_month="none",
                                      data_month="none", vintage="")
    return list(first_alarm.values())


def main(n_cal: int = 200) -> None:
    t0 = time.time()
    rows = []
    for name, ep in EPISODES.items():
        rows += episode_check(name, ep["peak"], ep["decisions"], n_cal, t0)
    df = pd.DataFrame(rows)
    df.to_csv("paper_assets/rd_realtime.csv", index=False)
    print(df.to_string(index=False))
    print(f"[{time.time()-t0:5.0f}s] wrote paper_assets/rd_realtime.csv")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
