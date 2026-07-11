"""m6x — generalized real-data application engine (design registered
in CHANGELOG, 2026-07-11 m6x entry; ILLUSTRATIVE per SPEC §4.5/§8).

Rolling causal monitoring of a FRED series with per-segment
parametric-bootstrap calibration (same machinery as m6_fred.py, which
is kept untouched). Additions over m6: pinned data snapshots under
data/ (live download only with --live), the lsc_tail_cusum detector,
and alarm ATTRIBUTION (composite: which feature's z crossed at the
alarm; tail: which arm).

Series (reference events fixed here, not chosen after results):
  indpro  monthly INDPRO log-growth, NBER peak months as events
  gdp     quarterly GDPC1 log-growth, NBER peaks + Great Moderation
          1984Q1 (McConnell-Perez-Quiros 2000) as a QUIETING event
  gs10    monthly 10y Treasury yield changes; Volcker 1979-10
          (vol-up) and ZLB 2008-12 (quieting) as events

Usage:
  python experiments/real_data.py SERIES [n_cal] [--train N]
         [--monitor N] [--far F] [--tag TAG] [--live]
Outputs: paper_assets/rd_{series}{tag}_alarms.csv / _summary.csv /
_figure.png (alarms.csv has an attribution column).
"""
from __future__ import annotations

import argparse
import glob
import io
import time
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lsc.dgp import AR1StateDGP
from lsc.diagnostics.alarms import calibrate, composite_score
from lsc.diagnostics.features import (
    compute_features,
    tail_exceedance,
    tail_shortfall,
)
from lsc.benchmarks.variance import raw_var_arm_at
from lsc.eval.detectors import (
    make_composite_detector,
    make_innovation_cusum_detector,
    make_raw_cusum_detector,
    make_raw_var_cusum_detector,
    make_tail_cusum_detector,
)
from lsc.models import KalmanModel

NBER_PEAKS = [
    "1948-11", "1953-07", "1957-08", "1960-04", "1969-12", "1973-11",
    "1980-01", "1981-07", "1990-07", "2001-03", "2007-12", "2020-02",
]
NBER_TROUGHS = [
    "1949-10", "1954-05", "1958-04", "1961-02", "1970-11", "1975-03",
    "1980-07", "1982-11", "1991-03", "2001-11", "2009-06", "2020-04",
]

SERIES = {
    "indpro": dict(
        fred_id="INDPRO", transform="log_growth", start="1948-01-01",
        months_per_obs=1, n_train=120, n_monitor=60,
        events=NBER_PEAKS, event_label="NBER peak",
        shade=list(zip(NBER_PEAKS, NBER_TROUGHS))),
    "gdp": dict(
        fred_id="GDPC1", transform="log_growth", start="1947-01-01",
        months_per_obs=3, n_train=60, n_monitor=20,
        events=NBER_PEAKS + ["1984-01"], event_label="NBER peak / GM84",
        shade=list(zip(NBER_PEAKS, NBER_TROUGHS))),
    "gs10": dict(
        fred_id="GS10", transform="diff", start="1953-04-01",
        months_per_obs=1, n_train=120, n_monitor=60,
        events=["1979-10", "2008-12"], event_label="vol-regime event",
        shade=[("1979-10", "1982-11")]),
}

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"


def load_series(cfg: dict, live: bool) -> pd.Series:
    snaps = sorted(glob.glob(f"data/{cfg['fred_id']}_*.csv"))
    if snaps and not live:
        raw = Path(snaps[-1]).read_text()
        src = snaps[-1]
    else:
        with urllib.request.urlopen(
                FRED_URL.format(id=cfg["fred_id"]), timeout=60) as r:
            raw = r.read().decode()
        src = "live"
    df = pd.read_csv(io.StringIO(raw), parse_dates=["observation_date"])
    s = df.set_index("observation_date")[cfg["fred_id"]].astype(float)
    if cfg["transform"] == "log_growth":
        s = 100.0 * np.log(s).diff().dropna()
    elif cfg["transform"] == "diff":
        s = s.diff().dropna()
    s = s[s.index >= cfg["start"]]
    print(f"loaded {cfg['fred_id']} from {src}: {len(s)} obs "
          f"({s.index[0]:%Y-%m} .. {s.index[-1]:%Y-%m})")
    return s


def fitted_null(Y_train: np.ndarray) -> AR1StateDGP:
    m = KalmanModel("ar1").fit(Y_train)
    p = dict(zip(m._param_names, m._params))
    phi = float(np.clip(p["ar.L1"], 0.01, 0.99))
    return AR1StateDGP(phi=phi, q=max(float(p["sigma2.ar"]), 1e-8),
                       r=max(float(p["sigma2.irregular"]), 1e-8))


def composite_attribution(score_fn, Y: np.ndarray, t: int) -> str:
    """Name of the feature whose standardized z is maximal at alarm t."""
    est = score_fn.model_factory().fit_filter(Y, n_train=score_fn.n_train)
    feats = compute_features(est, window=score_fn.window,
                             n_train=score_fn.n_train)
    best, best_z = "?", -np.inf
    for name in score_fn.include:
        c, s = score_fn.scales[name]
        z = abs(feats[name][t] - c[t]) / s[t]
        if np.isfinite(z) and z > best_z:
            best, best_z = name, z
    return best


def tail_attribution(Y: np.ndarray, n_train: int, t: int) -> str:
    est = KalmanModel("ar1").fit_filter(Y, n_train=n_train)
    up = tail_exceedance(est.innovations, n_train)[t]
    down = tail_shortfall(est.innovations, n_train)[t]
    return "exceedance_up" if np.nan_to_num(up) >= np.nan_to_num(down) \
        else "shortfall_down"


def run(series: str, n_cal: int, n_train: int | None, n_monitor: int | None,
        far: float, tag: str, live: bool) -> None:
    t0 = time.time()
    cfg = dict(SERIES[series])
    if n_train:
        cfg["n_train"] = n_train
    if n_monitor:
        cfg["n_monitor"] = n_monitor
    g = load_series(cfg, live)
    NT, NM = cfg["n_train"], cfg["n_monitor"]
    T_seg = NT + NM
    name = f"rd_{series}{tag}"

    alarms, seg_id = [], 0
    for start in range(0, len(g) - T_seg + 1, NM):
        seg = g.iloc[start:start + T_seg]
        Y = seg.values
        null = fitted_null(Y[:NT])
        comp_fn = make_composite_detector(
            lambda: KalmanModel("ar1"), null, T_seg, NT, n_scale_reps=50)
        detectors = {
            "lsc_composite": comp_fn,
            "lsc_tail_cusum": make_tail_cusum_detector(
                lambda: KalmanModel("ar1"), NT),
            "lsc_kalman_cusum": make_innovation_cusum_detector(
                lambda: KalmanModel("ar1"), NT),
            "raw_cusum": make_raw_cusum_detector(NT),
            "raw_var_cusum": make_raw_var_cusum_detector(NT),
        }
        for mname, fn in detectors.items():
            det = calibrate(mname, fn, null, T_seg, n_reps=n_cal,
                            far=far, seed0=100_000 + 1000 * seg_id)
            at = det.alarm_time(Y)
            if at is not None:
                if mname == "lsc_composite":
                    attr = composite_attribution(comp_fn, Y, at)
                elif mname == "lsc_tail_cusum":
                    attr = tail_attribution(Y, NT, at)
                elif mname == "raw_var_cusum":
                    attr = raw_var_arm_at(Y, NT, at)
                else:
                    attr = ""
                alarms.append(dict(segment=seg_id, method=mname,
                                   date=seg.index[at], feature=attr))
        print(f"[{time.time()-t0:5.0f}s] segment {seg_id} "
              f"({seg.index[NT]:%Y-%m} monitored)", flush=True)
        seg_id += 1

    adf = pd.DataFrame(alarms)
    adf.to_csv(f"paper_assets/{name}_alarms.csv", index=False)

    months_per = cfg["months_per_obs"]
    horizon = 12  # months
    events = pd.to_datetime([e + "-01" for e in cfg["events"]])
    rows = []
    methods = adf.method.unique() if len(adf) else []
    for method in methods:
        dates = pd.DatetimeIndex(sorted(adf[adf.method == method].date))
        for ev in events:
            after = dates[(dates >= ev)
                          & (dates <= ev + pd.DateOffset(months=horizon * 2))]
            rows.append(dict(
                method=method, event=f"{ev:%Y-%m}",
                first_alarm=(f"{after[0]:%Y-%m}" if len(after) else "none"),
                delay_months=(int((after[0].year - ev.year) * 12
                                  + after[0].month - ev.month)
                              if len(after) else np.nan)))
    summ = pd.DataFrame(rows)
    summ.to_csv(f"paper_assets/{name}_summary.csv", index=False)
    if len(summ):
        print(summ.pivot_table(index="event", columns="method",
                               values="delay_months").to_string())
    meta = dict(series=series, n_train=NT, n_monitor=NM, far=far,
                n_segments=seg_id, n_cal=n_cal,
                monitored_obs=seg_id * NM, months_per_obs=months_per)
    pd.DataFrame([meta]).to_csv(f"paper_assets/{name}_meta.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(g.index, g.values, lw=0.5, color="gray", label=cfg["fred_id"])
    for a, b in cfg["shade"]:
        ax.axvspan(pd.Timestamp(a + "-01"), pd.Timestamp(b + "-01"),
                   color="red", alpha=0.15)
    colors = {"lsc_composite": "C0", "lsc_tail_cusum": "C1",
              "lsc_kalman_cusum": "C2", "raw_cusum": "C3",
              "raw_var_cusum": "C4"}
    offs = {"lsc_composite": 1.0, "lsc_tail_cusum": 0.93,
            "lsc_kalman_cusum": 0.86, "raw_cusum": 0.79,
            "raw_var_cusum": 0.72}
    for method, gr in adf.groupby("method"):
        for d in gr.date:
            ax.plot([d], [ax.get_ylim()[1] * offs[method]], marker="v",
                    color=colors[method], ms=6)
        ax.plot([], [], marker="v", ls="", color=colors[method], label=method)
    ax.legend(fontsize=8)
    ax.set_title(f"{cfg['fred_id']} — causal alarms ({far:.0%} FAR per "
                 f"{NM}-obs window, parametric bootstrap) vs "
                 f"{cfg['event_label']} (shaded). Illustrative only.")
    fig.tight_layout()
    fig.savefig(f"paper_assets/{name}_figure.png", dpi=130)
    print(f"[{time.time()-t0:5.0f}s] wrote paper_assets/{name}_*")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("series", choices=list(SERIES))
    ap.add_argument("n_cal", nargs="?", type=int, default=200)
    ap.add_argument("--train", type=int, default=None)
    ap.add_argument("--monitor", type=int, default=None)
    ap.add_argument("--far", type=float, default=0.05)
    ap.add_argument("--tag", default="")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    run(a.series, a.n_cal, a.train, a.monitor, a.far, a.tag, a.live)
