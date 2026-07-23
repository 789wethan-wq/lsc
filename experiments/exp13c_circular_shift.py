"""exp13c_circular_shift.py -- a genuinely dependence-aware joint FWER
test, built to replace the failed Tippett/min-p attempt in
exp13_joint_fwer.py.

WHY THIS ONE WORKS WHERE THE FIRST ATTEMPT DIDN'T: exp13_joint_fwer.py's
null redrew each method's alarm months INDEPENDENTLY within a shared
draw -- which doesn't model real cross-method correlation at all, just
runs five independent tests and combines them (closer to a Sidak/
Bonferroni combination in different clothing). This script instead
shifts ALL FIVE methods' alarm months by the SAME random amount s in
each draw. A rigid common shift preserves every pairwise relative
offset between methods exactly -- whatever real correlation exists in
when different methods fire (because they share underlying CUSUM
machinery on the same series) survives the shift untouched. What the
shift destroys is alignment with the FIXED event calendar, which is
exactly the null hypothesis we want: "would this alarm configuration,
exactly as internally correlated as it really is, hit the real event
calendar this well by chance under an arbitrary rotation of the
calendar?"

FIX (this version): the shift must wrap alarms within the TRUE absolute
calendar bounds of the monitored window -- [window_start_idx,
window_start_idx + n_months - 1] -- not an arbitrary [0, n_months)
space computed from a fixed 1948-01 epoch that has no necessary
relationship to where the real monitored window actually sits. An
earlier version used `(a + s) % n_months` directly on absolute month
indices anchored at 1948-01; since INDPRO's monitored window starts at
1958-01 (absolute index 120, not 0), any event or alarm with absolute
index >= n_months (e.g. the 2020-02 COVID NBER peak, index 865 against
n_months=780) was STRUCTURALLY UNREACHABLE by any shifted alarm, for
any seed -- shifted alarms always land in [0, n_months), never above
it, regardless of s. Two of the seven observed hits in the original
INDPRO run were hits against exactly that unreachable event, silently
inflating the apparent significance (p=0.021-0.023 instead of the
corrected p=0.033-0.036 -- verified by comparison, both before and
after this fix; see paper_assets/exp13c_indpro_corrected.txt).

The corrected shift is `window_start_idx + ((a - window_start_idx + s)
% n_months)`: this always keeps shifted alarms within the window's true
absolute range, so any real event that occurred DURING the monitored
window (whenever in that window) is reachable, and any event entirely
outside the window (before it started) correctly remains unreachable
in both the observed and null computations, matching what "the
monitored data literally could never have alarmed near that event"
already means in reality -- not an artifact of the modulus arithmetic.
Event indices are never rebased or wrapped; only alarms are shifted,
exactly as before.

`window_start_idx` must be the ABSOLUTE month index (same to_month_index
convention, start=(1948,1)) of the first month actually monitored --
i.e. the first segment's test_start in real_data_date_boundaries.csv,
NOT the series' full data start. Get this per-series from that file;
do not assume it matches another series' value.

STATUS: run and verified for INDPRO only (its exact registered event
dates are available with full provenance from
paper_assets/exp13_indpro_series_data.json). GDP/GS10/UNRATE need the
same treatment, with their OWN window_start_idx pulled from
real_data_date_boundaries.csv -- do not assume this generalizes to
them without running it.

Usage: python experiments/exp13c_circular_shift.py [series_json] [n_perm]
Output: prints observed_hits/observed_total/p_total_hits/... to stdout.
"""
from __future__ import annotations

import json
import sys

import numpy as np


def to_month_index(datestr: str, start=(1948, 1)) -> int:
    y, m, _ = datestr.split("-")
    return (int(y) - start[0]) * 12 + (int(m) - start[1])


def hit_count(alarm_months, event_months, hit_window):
    hits = 0
    for e in event_months:
        if any(e <= a <= e + hit_window for a in alarm_months):
            hits += 1
    return hits


def circular_shift_joint_test(series: dict, n_perm: int = 20000, seed: int = 2026):
    n_months = series["n_months"]
    window_start = series["window_start_idx"]
    event_months = series["event_months"]
    hit_window = series["hit_window"]
    alarms = series["alarms"]
    methods = [m for m in alarms if len(alarms[m]) > 0]

    observed_hits = {m: hit_count(alarms[m], event_months, hit_window) for m in methods}
    observed_total = sum(observed_hits.values())
    observed_max = max(observed_hits.values())

    rng = np.random.default_rng(seed)
    null_total = np.empty(n_perm, dtype=int)
    null_max = np.empty(n_perm, dtype=int)
    for i in range(n_perm):
        s = rng.integers(0, n_months)
        total = 0
        mx = 0
        for m in methods:
            # shift within [window_start, window_start + n_months - 1] --
            # the window's TRUE absolute range -- not [0, n_months).
            shifted = [window_start + ((a - window_start + s) % n_months)
                       for a in alarms[m]]
            h = hit_count(shifted, event_months, hit_window)
            total += h
            mx = max(mx, h)
        null_total[i] = total
        null_max[i] = mx

    p_total = float((null_total >= observed_total).mean())
    p_max = float((null_max >= observed_max).mean())
    return dict(
        observed_hits=observed_hits, observed_total=observed_total,
        observed_max=observed_max, p_total_hits=p_total, p_max_single_method=p_max,
        null_total_mean=float(null_total.mean()), null_total_sd=float(null_total.std()),
    )


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "paper_assets/exp13_indpro_series_data.json"
    n_perm = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    with open(path) as f:
        raw = json.load(f)
    series = dict(
        n_months=raw["n_months"],
        window_start_idx=to_month_index(raw["window_start"]),
        hit_window=raw["hit_window"],
        event_months=[to_month_index(d) for d in raw["event_months"]],
        alarms={m: [to_month_index(d) for d in v] for m, v in raw["alarms"].items()},
    )
    out = circular_shift_joint_test(series, n_perm=n_perm)
    print(f"n_perm={n_perm}")
    for k, v in out.items():
        print(f"  {k}: {v}")
