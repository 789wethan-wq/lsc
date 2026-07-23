"""exp13d_export_other_series.py -- exports GDP, GS10, and UNRATE's
registered-event and per-method alarm-month data in the exact schema
already used by exp13_indpro_series_data.json, so exp13c_circular_shift.py
can run on them unmodified.

Reuses real_data_eval.py's own `monitored_months()` and event-filtering
logic directly (not reimplemented), so each series' event_months here
are exactly the `events_in` that actually produced Table 6's "Hits /
events" denominators -- not the raw, unfiltered SERIES[...]["events"]
list, which for INDPRO includes NBER peaks from before 1958 that were
never in the monitored window.

GDP hit_window resolved explicitly, not assumed: real_data_eval.py
defines `HORIZON_MONTHS = 12` as a single module-level constant and
applies it identically to every series via `hits_within()` -- there is
no quarter-native special-casing for GDP anywhere in that file. So
hit_window=12 (months, not quarters) is confirmed correct for GDP too,
not an unchanged carryover assumption from INDPRO.

GDP also needs `step=3` in its output JSON (exp13c_circular_shift.py's
shift granularity): GDPC1 is quarterly, so real GDP alarms can only
ever land on a quarterly grid, and a monthly-granularity shift moves
them onto calendar months GDP was never observed at -- checked
directly (not assumed) during review: constraining the shift to
quarters raised GDP's already-significant result from p=0.0082 to
p<0.00005, rather than lowering it, so this is not a conservatism
fudge -- it's the properly specified null. GS10/UNRATE/INDPRO are true
monthly series (alarm and event indices span all three mod-3 residues,
confirmed) and get the default step=1.

window_start AND window_end for each series come from
real_data_date_boundaries.csv (the first segment's test_start and last
segment's test_end) -- n_months is derived as the actual month-index
distance between them, rounded up to a multiple of `step`, NOT from
`len(monitored_months(...))`. That count is in units of MONITORED
OBSERVATIONS (n_segments * n_monitor), which is a month count for
monthly series but a QUARTER count for GDP -- using it directly as a
month-count for GDP understated its window by 3x (240 "months" when
the real window is ~718-720 months wide), silently truncating the
window in exactly the way the original INDPRO epoch bug did, just via
a different mechanism (a units mismatch instead of an anchor mismatch).
Checked directly: with the corrected n_months, every real event and
alarm index for all four series now falls inside
[window_start_idx, window_start_idx + n_months - 1].

Usage: python experiments/exp13d_export_other_series.py
Output: paper_assets/exp13_{gdp,gs10,unrate}_series_data.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from exp13c_circular_shift import to_month_index  # noqa: E402
from real_data import SERIES  # noqa: E402
from real_data_eval import monitored_months  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "paper_assets"


def export_series(series_name: str, out_path: Path) -> None:
    meta = pd.read_csv(ASSETS / f"rd_{series_name}_meta.csv").iloc[0]
    cfg = SERIES[series_name]

    # event_months: exactly real_data_eval.py's events_in -- the
    # SERIES config's raw event list, filtered to the monitored range.
    events = pd.to_datetime([e + "-01" for e in cfg["events"]])
    mon = monitored_months(series_name, meta)
    events_in = events[(events >= mon.min() - pd.DateOffset(months=1))
                       & (events <= mon.max())]
    event_months = [f"{d:%Y-%m-01}" for d in sorted(events_in)]

    # alarms: the real rd_{series}_alarms.csv, the exact file that
    # produced Table 6 (already reconstruction-verified against Table 6
    # for INDPRO in an earlier round).
    alarms: dict[str, list[str]] = {}
    with open(ASSETS / f"rd_{series_name}_alarms.csv") as f:
        for row in csv.DictReader(f):
            date = pd.Timestamp(row["date"])
            alarms.setdefault(row["method"], []).append(f"{date:%Y-%m-01}")
    for m in alarms:
        alarms[m].sort()

    # window_start / window_end: this series' OWN first-segment
    # test_start and last-segment test_end, from
    # real_data_date_boundaries.csv -- not assumed to match INDPRO's,
    # and NOT derived from len(monitored_months(...)), which counts
    # observations (quarters for GDP), not months.
    boundaries = pd.read_csv(ASSETS / "real_data_date_boundaries.csv")
    series_bounds = boundaries[boundaries.series == series_name].sort_values("segment")
    window_start = f"{series_bounds.iloc[0].test_start}-01"
    window_end = f"{series_bounds.iloc[-1].test_end}-01"

    step = 3 if series_name == "gdp" else 1  # GDPC1 is quarterly; see module docstring

    ws_idx = to_month_index(window_start)
    we_idx = to_month_index(window_end)
    true_width = we_idx - ws_idx + 1
    n_months = -(-true_width // step) * step  # round up to a multiple of step

    # Guard against exactly this bug class recurring silently: every
    # real event and alarm index must fall inside the stated window.
    all_idx = ([to_month_index(d) for d in event_months]
               + [to_month_index(d) for v in alarms.values() for d in v])
    out_of_window = [i for i in all_idx if not (ws_idx <= i <= ws_idx + n_months - 1)]
    if out_of_window:
        raise ValueError(
            f"{series_name}: {len(out_of_window)} event/alarm index(es) fall "
            f"outside the stated window [{ws_idx}, {ws_idx + n_months - 1}] -- "
            f"n_months is wrong, do not write this file")

    out = dict(
        n_months=n_months,
        window_start=window_start,
        step=step,
        event_months=event_months,
        hit_window=12,
        alarms=alarms,
    )
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}: n_months={n_months}, window_start={window_start}, "
          f"step={step}, {len(event_months)} events, "
          f"{sum(len(v) for v in alarms.values())} total alarms across "
          f"{len(alarms)} methods -- window bounds check passed "
          f"({len(all_idx)} indices, all inside)")


if __name__ == "__main__":
    for series in ["gdp", "gs10", "unrate"]:
        export_series(series, ASSETS / f"exp13_{series}_series_data.json")
