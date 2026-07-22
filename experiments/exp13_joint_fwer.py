"""exp13 -- joint, per-series family-wise error control across the five
real-data methods (referee-suggested tightening, external review this
round). The Table 6 multiple-comparisons correction (Bonferroni/BH-FDR
across all 19 method x series tests) treats every test as independent.
That is valid (Bonferroni does not require independence) but
conservative, since the five methods on a given series are computed
from overlapping CUSUM machinery on the same underlying path and are
plausibly positively correlated in *when* they fire.

This runs the REAL per-method alarm-month arrays and NBER/registered
event-month arrays behind Table 6 (from paper_assets/rd_{series}_alarms.csv,
rd_{series}_meta.csv, and the real_data/real_data_eval monitored-month
and event machinery -- the exact functions the paper's own permutation
test uses, imported directly, not reimplemented) through a per-series
Tippett (min-p) combination test: for each series, combine the five
methods' hit-counts into one statistic (the smallest per-method
p-value), and build that statistic's null by, on each joint draw,
redrawing every method's alarm months independently (respecting each
method's own observed alarm count) from the SAME monitored-month
universe -- so the correlation the five methods share through that
common universe (though not through correlated *which*-month draws,
which would need the real per-method score paths) is preserved in the
null by construction, rather than assumed away. A final Bonferroni
step (n=4) controls FWER across the four series.

Usage: python experiments/exp13_joint_fwer.py [n_perm]
Output: paper_assets/exp13_joint_fwer.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from real_data import SERIES  # noqa: E402
from real_data_eval import hits_within, monitored_months  # noqa: E402

SEED = 20260720


def hit_count(alarm_dates: pd.DatetimeIndex, events: pd.DatetimeIndex) -> int:
    return hits_within(alarm_dates, events)


def load_series_inputs(series_key: str) -> dict:
    meta = pd.read_csv(f"paper_assets/rd_{series_key}_meta.csv").iloc[0]
    mon = monitored_months(series_key, meta)
    cfg = SERIES[series_key]
    events = pd.to_datetime([e + "-01" for e in cfg["events"]])
    events_in = events[(events >= mon.min() - pd.DateOffset(months=1))
                       & (events <= mon.max())]
    adf = pd.read_csv(f"paper_assets/rd_{series_key}_alarms.csv",
                      parse_dates=["date"])
    alarms = {m: pd.DatetimeIndex(sorted(adf[adf.method == m].date))
             for m in sorted(adf.method.unique())}
    return dict(n_months=len(mon), months=mon, events=events_in,
               alarms=alarms)


def method_null_hitcounts(n_alarms: int, months: pd.DatetimeIndex,
                          events: pd.DatetimeIndex, n_ref: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Reference null distribution of hit-counts for one method (alarm
    months resampled uniformly w/o replacement from the monitored
    universe, respecting the method's own alarm count)."""
    months_arr = months.values
    out = np.empty(n_ref, dtype=int)
    for i in range(n_ref):
        draw = pd.DatetimeIndex(rng.choice(months_arr, size=n_alarms,
                                           replace=False))
        out[i] = hit_count(draw, events)
    return out


def joint_series_p(series: dict, n_perm: int, n_ref: int,
                   seed: int) -> dict:
    rng = np.random.default_rng(seed)
    months, events = series["months"], series["events"]
    alarms = series["alarms"]
    methods = [m for m, dts in alarms.items() if len(dts) > 0]
    if not methods:
        return None

    n_alarms = {m: len(alarms[m]) for m in methods}
    observed_hits = {m: hit_count(alarms[m], events) for m in methods}

    # reference null hit-count distribution per method (for converting
    # any hit-count, observed or under the joint null, into a p-value
    # on a common, comparable scale)
    ref_null = {m: method_null_hitcounts(n_alarms[m], months, events,
                                        n_ref, rng) for m in methods}
    observed_p = {m: float((ref_null[m] >= observed_hits[m]).mean())
                 for m in methods}
    observed_minp = min(observed_p.values())

    # joint null: each draw redraws ALL methods' alarm months from the
    # same monitored universe; convert each method's draw hit-count to
    # a p-value via its OWN reference null distribution (fixes the
    # placeholder-indicator bug in the original spec: hit-counts must
    # be converted to a comparable p-value, not compared as raw booleans)
    null_minp = np.empty(n_perm)
    months_arr = months.values
    for i in range(n_perm):
        ps = []
        for m in methods:
            draw = pd.DatetimeIndex(rng.choice(months_arr, size=n_alarms[m],
                                               replace=False))
            h = hit_count(draw, events)
            ps.append(float((ref_null[m] >= h).mean()))
        null_minp[i] = min(ps)

    joint_p = float((null_minp <= observed_minp).mean())
    return dict(methods=methods, n_alarms=n_alarms,
               observed_hits=observed_hits, observed_p=observed_p,
               observed_minp=observed_minp, joint_p=joint_p)


def main(n_perm: int = 20000, n_ref: int = 20000) -> None:
    rows = []
    for i, key in enumerate(("indpro", "gdp", "gs10", "unrate")):
        series = load_series_inputs(key)
        out = joint_series_p(series, n_perm=n_perm, n_ref=n_ref,
                             seed=SEED + i)
        joint_p_bonf = min(1.0, out["joint_p"] * 4)
        print(f"=== {key} ===")
        print(f"  n_months={series['n_months']}, n_events={len(series['events'])}")
        for m in out["methods"]:
            print(f"  {m:18s} n_alarms={out['n_alarms'][m]:2d} "
                  f"hits={out['observed_hits'][m]} "
                  f"observed_p={out['observed_p'][m]:.4f}")
        print(f"  observed min-p (Tippett) = {out['observed_minp']:.4f}")
        print(f"  joint per-series p (min-p null, n_perm={n_perm}) = "
              f"{out['joint_p']:.4f}")
        print(f"  joint p, Bonferroni x4 across series = {joint_p_bonf:.4f}")
        rows.append(dict(series=key, n_months=series["n_months"],
                         n_events=len(series["events"]),
                         observed_minp=round(out["observed_minp"], 4),
                         minp_method=min(out["observed_p"], key=out["observed_p"].get),
                         joint_p=round(out["joint_p"], 4),
                         joint_p_bonferroni_x4=round(joint_p_bonf, 4)))

    df = pd.DataFrame(rows)
    df.to_csv("paper_assets/exp13_joint_fwer.csv", index=False)
    print("\nwrote paper_assets/exp13_joint_fwer.csv")


if __name__ == "__main__":
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    n_ref = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    main(n_perm, n_ref)
