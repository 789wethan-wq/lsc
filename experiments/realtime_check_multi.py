"""Generalized real-time (ALFRED) vintage check (SPEC R5 M3,
pre-registered in experiments/CHANGELOG.md 2026-07-26 BEFORE this
script was run) -- extends `realtime_check.py`'s INDPRO-only protocol
to GS10 and UNRATE (monthly, n_train=120/n_monitor=60, identical
window lengths to INDPRO's -- a mechanical extension). GDP is
deliberately NOT attempted here: it is quarterly (n_train=60/
n_monitor=20 per real_data.py's own SERIES config, a materially
different decision-month grid, not a parameter swap), and Sec 9's
existing text already explains why rushing this exact kind of
extension is a documented bug risk (a window-anchoring error and a
GDP quarter/month units mismatch, both from previous rolling-window
protocol extensions) -- deferred as its own separately-scoped
follow-up, per the user's explicit confirmation.

`realtime_check.py` itself is left untouched (same convention as
real_data.py being the generalized engine while m6_fred.py stays
untouched) -- this is a NEW script, parameterized by series config
(fred_id, transform, episodes) instead of INDPRO's hardcoded values,
reusing the identical per-decision-month protocol (train on the 120
months ending at the reference date, monitor from the month after,
parametric-bootstrap recalibration per vintage, first-alarm month +
data-month crossing).

VERIFICATION (run before trusting any new series): this script is
first run with series="indpro" and checked against the published
paper_assets/rd_realtime.csv -- if it does not reproduce INDPRO's
existing gfc/covid alarm months exactly, the generalization has a bug
and GS10/UNRATE output should not be trusted until it does.

Episodes chosen from each series' OWN existing event list
(experiments/real_data.py's SERIES dict), not new events:
  unrate  reuses INDPRO's gfc (peak 2007-12) / covid (peak 2020-02)
          episodes -- UNRATE's own event list IS NBER_PEAKS, and these
          are the two most narratively load-bearing (they are what the
          abstract/Sec 9 already foreground for INDPRO).
  gs10    its own three events: 1979-10 (Volcker, vol-up), 2008-12
          (ZLB, quieting), 2022-03 (rate-hike vol event) -- genuinely
          different from NBER peaks, so tested on GS10's own dates
          rather than reused ones. Decision windows sized the same way
          as gfc/covid (18/10 months): 1979-10 gets an 18-month window
          (matching gfc's), 2008-12 and 2022-03 get 10-month windows
          (matching covid's) -- there is no NBER announcement lag to
          size these against, so this is a disclosed judgment call,
          not a derived quantity.

Usage: python experiments/realtime_check_multi.py SERIES [n_cal]
       (SERIES in {indpro, gs10, unrate}; indpro is the verification run)
Output: paper_assets/rd_realtime_{series}.csv
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
    make_raw_var_cusum_detector,
    make_tail_cusum_detector,
)
from lsc.models import KalmanModel

ALFRED = "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={fred_id}&vintage_date={d}"
N_TRAIN, N_MONITOR = 120, 60
FAR = 0.05

SERIES_CONFIG = {
    "indpro": dict(fred_id="INDPRO", transform="log_growth", start="1948-01-01"),
    "gs10": dict(fred_id="GS10", transform="diff", start="1953-04-01"),
    "unrate": dict(fred_id="UNRATE", transform="diff", start="1948-01-01"),
}

EPISODES = {
    "indpro": {
        "gfc": dict(peak="2007-12", decisions=pd.period_range("2008-01", "2009-06", freq="M")),
        "covid": dict(peak="2020-02", decisions=pd.period_range("2020-03", "2020-12", freq="M")),
    },
    "unrate": {
        "gfc": dict(peak="2007-12", decisions=pd.period_range("2008-01", "2009-06", freq="M")),
        "covid": dict(peak="2020-02", decisions=pd.period_range("2020-03", "2020-12", freq="M")),
    },
    "gs10": {
        # "volcker" (1979-10) is NOT included: direct ALFRED queries
        # (2026-07-26) confirm GS10's vintage history does not extend
        # that far back -- 404 at 1979/1990/1994/1996-06, first 200 at
        # 1997-01. A genuine data-availability limit, not a bug; the
        # earlier "-checked 2026-07-23: ALFRED serves vintage histories
        # for all three series" note in Sec 9 confirmed EXISTENCE, not
        # coverage back to 1979 specifically -- corrected here.
        "zlb": dict(peak="2008-12", decisions=pd.period_range("2009-01", "2009-10", freq="M")),
        "hike2022": dict(peak="2022-03", decisions=pd.period_range("2022-04", "2023-01", freq="M")),
    },
}


def load_vintage(fred_id: str, transform: str, vintage_day: str) -> pd.Series:
    cache = Path(f"data/vintages/{fred_id}_{vintage_day}.csv")
    if cache.exists():
        raw = cache.read_text()
    else:
        with urllib.request.urlopen(
                ALFRED.format(fred_id=fred_id, d=vintage_day), timeout=60) as r:
            raw = r.read().decode()
        cache.write_text(raw)
    df = pd.read_csv(io.StringIO(raw), parse_dates=["observation_date"])
    col = [c for c in df.columns if c.startswith(fred_id)][0]
    s = df.set_index("observation_date")[col].replace(".", np.nan)
    s = s.astype(float).dropna()
    if transform == "log_growth":
        g = 100.0 * np.log(s).diff().dropna()
    elif transform == "diff":
        g = s.diff().dropna()
    else:
        raise ValueError(f"unknown transform: {transform}")
    return g


def fitted_null(Y_train: np.ndarray) -> AR1StateDGP:
    m = KalmanModel("ar1").fit(Y_train)
    p = dict(zip(m._param_names, m._params))
    phi = float(np.clip(p["ar.L1"], 0.01, 0.99))
    return AR1StateDGP(phi=phi, q=max(float(p["sigma2.ar"]), 1e-8),
                       r=max(float(p["sigma2.irregular"]), 1e-8))


def episode_check(series: str, fred_id: str, transform: str, start: str,
                  name: str, peak: str, decisions, n_cal: int, t0: float) -> list[dict]:
    peak_ts = pd.Timestamp(peak + "-01")
    T_cal = N_TRAIN + N_MONITOR
    first_alarm: dict[str, dict] = {}
    for m in decisions:
        vday = f"{m.year}-{m.month:02d}-15"
        g = load_vintage(fred_id, transform, vday)
        g = g[g.index >= start]
        train = g[g.index <= peak_ts].iloc[-N_TRAIN:]
        monitor = g[g.index > peak_ts]
        if len(train) < N_TRAIN or len(monitor) == 0:
            continue
        Y = np.concatenate([train.values, monitor.values])
        idx = train.index.append(monitor.index)
        Y_pad = np.concatenate([Y, np.zeros(T_cal - len(Y))]) \
            if len(Y) < T_cal else Y[:T_cal]
        null = fitted_null(train.values)
        detectors = {
            "lsc_composite": make_composite_detector(
                lambda: KalmanModel("ar1"), null, T_cal, N_TRAIN, n_scale_reps=50),
            "lsc_tail_cusum": make_tail_cusum_detector(
                lambda: KalmanModel("ar1"), N_TRAIN),
            "raw_cusum": make_raw_cusum_detector(N_TRAIN),
            "raw_var_cusum": make_raw_var_cusum_detector(N_TRAIN),
        }
        for mname, fn in detectors.items():
            if mname in first_alarm:
                continue
            det = calibrate(mname, fn, null, T_cal, n_reps=n_cal, far=FAR, seed0=100_000)
            at = det.alarm_time(Y_pad)
            if at is not None and at < len(Y):
                first_alarm[mname] = dict(
                    series=series, episode=name, method=mname,
                    decision_month=str(m), data_month=f"{idx[at]:%Y-%m}", vintage=vday)
        print(f"[{time.time()-t0:5.0f}s] {series}/{name} decision {m}: "
              f"alarmed so far: {sorted(first_alarm)}", flush=True)
        if len(first_alarm) == 4:
            break
    for mname in ("lsc_composite", "lsc_tail_cusum", "raw_cusum", "raw_var_cusum"):
        if mname not in first_alarm:
            first_alarm[mname] = dict(series=series, episode=name, method=mname,
                                      decision_month="none", data_month="none", vintage="")
    return list(first_alarm.values())


def main(series: str, n_cal: int = 200) -> None:
    t0 = time.time()
    cfg = SERIES_CONFIG[series]
    rows = []
    for name, ep in EPISODES[series].items():
        rows += episode_check(series, cfg["fred_id"], cfg["transform"], cfg["start"],
                              name, ep["peak"], ep["decisions"], n_cal, t0)
    df = pd.DataFrame(rows)
    out_path = f"paper_assets/rd_realtime_{series}.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"[{time.time()-t0:5.0f}s] wrote {out_path}")


if __name__ == "__main__":
    series = sys.argv[1] if len(sys.argv) > 1 else "indpro"
    n_cal = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    main(series, n_cal)
