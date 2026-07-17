"""m6x evaluation — false-alarm accounting and permutation tests for
the real-data alarm sets (design registered in CHANGELOG m6x entry).

For every rd_* run found in paper_assets/:
  - hits: reference events with >= 1 alarm in (event, event + 12 months]
  - expected stray alarms under the null: n_segments * FAR
  - observed non-event alarms (not within 12 months after any event)
  - permutation p-value: resample the observed number of alarm months
    uniformly (without replacement) from all monitored months, 20,000
    draws; p = P(perm hits >= observed hits). Small p = alarms cluster
    after reference events more than chance.

Output: paper_assets/rd_eval.csv (+ printed table).
"""
from __future__ import annotations

import glob
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from real_data import SERIES, load_series  # noqa: E402

HORIZON_MONTHS = 12
N_PERM = 20_000


def monitored_months(series_key: str, meta: pd.Series) -> pd.DatetimeIndex:
    cfg = SERIES[series_key]
    g = load_series(cfg, live=False)
    NT, NM = int(meta.n_train), int(meta.n_monitor)
    idx = []
    seg = 0
    for start in range(0, len(g) - (NT + NM) + 1, NM):
        idx.extend(g.index[start + NT: start + NT + NM])
        seg += 1
    return pd.DatetimeIndex(idx)


def hits_within(alarm_dates: pd.DatetimeIndex,
                events: pd.DatetimeIndex) -> int:
    n = 0
    for ev in events:
        hi = ev + pd.DateOffset(months=HORIZON_MONTHS)
        if ((alarm_dates >= ev) & (alarm_dates <= hi)).any():
            n += 1
    return n


def seeded_rng(run_name: str, method: str) -> np.random.Generator:
    """Deterministic per-(run, method) RNG, independent of how many
    other rd_* runs exist or what order glob returns them in — a
    shared, sequentially-consumed RNG made every run's p-value depend
    on unrelated files added to paper_assets/ (P2 fix, 2026-07-16:
    caught because rd_indpro's own permutation p-value shifted
    0.0073 -> 0.0092 after adding unrelated rd_unrate/_far1/_far20
    runs, even though rd_indpro's alarm data was byte-identical)."""
    digest = zlib.crc32(f"{run_name}:{method}".encode())
    return np.random.default_rng(np.random.SeedSequence([20260711, digest]))


def main() -> None:
    rows = []
    for meta_path in sorted(glob.glob("paper_assets/rd_*_meta.csv")):
        run_name = re.sub(r"_meta\.csv$", "", Path(meta_path).name)
        meta = pd.read_csv(meta_path).iloc[0]
        series_key = meta.series
        cfg = SERIES[series_key]
        events = pd.to_datetime([e + "-01" for e in cfg["events"]])
        adf = pd.read_csv(f"paper_assets/{run_name}_alarms.csv",
                          parse_dates=["date"]) \
            if Path(f"paper_assets/{run_name}_alarms.csv").stat().st_size > 30 \
            else pd.DataFrame(columns=["method", "date"])
        mon = monitored_months(series_key, meta)
        events_in = events[(events >= mon.min() - pd.DateOffset(months=1))
                           & (events <= mon.max())]
        for method in sorted(set(adf.method)) if len(adf) else []:
            dts = pd.DatetimeIndex(sorted(adf[adf.method == method].date))
            obs_hits = hits_within(dts, events_in)
            # alarms that are near no event = stray (false-alarm budget)
            near_any = np.zeros(len(dts), dtype=bool)
            for ev in events_in:
                near_any |= np.asarray(
                    (dts >= ev) & (dts <= ev + pd.DateOffset(
                        months=HORIZON_MONTHS)))
            n_stray = int((~near_any).sum())
            # permutation (seeded per run+method: see seeded_rng)
            rng = seeded_rng(run_name, method)
            perm_hits = np.empty(N_PERM)
            mon_arr = mon.values
            for b in range(N_PERM):
                samp = pd.DatetimeIndex(
                    rng.choice(mon_arr, size=len(dts), replace=False))
                perm_hits[b] = hits_within(samp, events_in)
            pval = float((perm_hits >= obs_hits).mean())
            rows.append(dict(
                run=run_name, method=method, far=meta.far,
                n_segments=int(meta.n_segments),
                expected_stray=round(float(meta.n_segments * meta.far), 2),
                n_alarms=len(dts), n_events=len(events_in),
                hits=obs_hits, stray_alarms=n_stray,
                perm_mean_hits=round(float(perm_hits.mean()), 2),
                perm_p=round(pval, 4)))
            print(f"{run_name:22s} {method:18s} alarms={len(dts):2d} "
                  f"hits={obs_hits}/{len(events_in)} stray={n_stray} "
                  f"(exp {meta.n_segments * meta.far:.1f}) p={pval:.4f}",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("paper_assets/rd_eval.csv", index=False)
    print("wrote paper_assets/rd_eval.csv")


if __name__ == "__main__":
    main()
