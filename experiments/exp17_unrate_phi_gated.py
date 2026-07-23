"""exp17_unrate_phi_gated.py -- a pre-specified model-fit gate for
UNRATE's marginal real-data association (raw_cusum / lsc_kalman_cusum,
Table 6), per the review's Question 4: rather than reporting the
phi-clipping confound only qualitatively (as Sec 9 currently does),
recompute the permutation p-value with clipped-phi windows excluded
from BOTH the hit count and the resampling universe -- so the gated
test is apples-to-apples, not "gated hits against an ungated null."

WHAT'S ALREADY KNOWN (verified against real data): raw_cusum and
lsc_kalman_cusum's 4 hits each fall one-per-segment in segments 3
(1974-11, phi=0.01 clipped), 8 (2001, phi=0.01 clipped), 10 (2008 GFC,
phi=0.01 clipped), and 12 (2020, phi=0.948 unclipped). Excluding the
three clipped segments therefore drops both detectors from 4/9 to 1/9
hits (segment 12's COVID hit only) -- but the RIGHT comparison isn't
"1/9 against the old 9-event, unrestricted-resampling null." The
null's monitored-months universe must ALSO exclude the clipped
segments' months, or the test compares a restricted numerator against
an unrestricted denominator.

Per-segment phi comes directly from
paper_assets/exp09_ljungbox_table.csv (already computed by
experiments/exp09_real_data_fit_check.py, not refit here) -- that
table's `phi` column is already the CLIPPED value used by the actual
pipeline, so the clip_bounds check (phi in (0.01, 0.99) strictly) is
exactly "did the pipeline's own clip bind for this segment," not a
re-derivation of the unclipped MLE.

Per-segment monitored months come directly from
paper_assets/real_data_date_boundaries.csv's test_start/test_end per
segment (already computed by experiments/real_data_date_boundaries.py).

Usage: python experiments/exp17_unrate_phi_gated.py [n_perm]
Output: paper_assets/exp17_unrate_phi_gated.csv (+ printed table)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from exp13c_circular_shift import to_month_index  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "paper_assets"


def build_gated_universe(segment_phi: dict, monitored_months_by_segment: dict,
                          clip_bounds=(0.01, 0.99)) -> tuple[set, set]:
    """Returns (unclipped_months, unclipped_segment_ids) -- the months
    eligible for both the hit count and the resampling null once
    clipped-phi segments are excluded."""
    unclipped_segments = {
        seg for seg, phi in segment_phi.items()
        if clip_bounds[0] < phi < clip_bounds[1]
    }
    unclipped_months = set()
    for seg in unclipped_segments:
        unclipped_months.update(monitored_months_by_segment[seg])
    return unclipped_months, unclipped_segments


def gated_permutation_test(alarms: Sequence[int], events: Sequence[int],
                            unclipped_months: set, hit_window: int,
                            n_perm: int = 20000, seed: int = 2026) -> dict:
    import numpy as np
    rng = np.random.default_rng(seed)

    gated_alarms = [a for a in alarms if a in unclipped_months]
    gated_events = [e for e in events if e in unclipped_months]

    def hit_count(alarm_months, event_months):
        return sum(1 for e in event_months
                   if any(e <= a <= e + hit_window for a in alarm_months))

    observed = hit_count(gated_alarms, gated_events)
    universe = sorted(unclipped_months)
    if len(universe) < len(gated_alarms):
        raise ValueError("gated universe smaller than the number of gated "
                          "alarms -- resampling without replacement would "
                          "be ill-defined; check the universe construction.")

    null_hits = np.empty(n_perm, dtype=int)
    for i in range(n_perm):
        resampled = rng.choice(universe, size=len(gated_alarms), replace=False)
        null_hits[i] = hit_count(resampled, gated_events)

    p = float((null_hits >= observed).mean())
    return dict(gated_observed_hits=observed, gated_n_events=len(gated_events),
                gated_n_alarms=len(gated_alarms), p_gated=p,
                null_mean=float(null_hits.mean()))


def main(n_perm: int = 20000) -> None:
    phi_df = pd.read_csv(ASSETS / "exp09_ljungbox_table.csv")
    phi_df = phi_df[phi_df.series == "unrate"]
    segment_phi = dict(zip(phi_df.segment, phi_df.phi))

    boundaries = pd.read_csv(ASSETS / "real_data_date_boundaries.csv")
    boundaries = boundaries[boundaries.series == "unrate"]
    monitored_months_by_segment = {}
    for _, row in boundaries.iterrows():
        start_idx = to_month_index(f"{row.test_start}-01")
        end_idx = to_month_index(f"{row.test_end}-01")
        monitored_months_by_segment[row.segment] = list(range(start_idx, end_idx + 1))

    unclipped_months, unclipped_segments = build_gated_universe(
        segment_phi, monitored_months_by_segment)
    clipped_segments = sorted(set(segment_phi) - unclipped_segments)
    print(f"segments: {sorted(segment_phi)}")
    print(f"clipped (phi at [0.01,0.99] boundary): {clipped_segments} "
          f"(phi={[segment_phi[s] for s in clipped_segments]})")
    print(f"unclipped: {sorted(unclipped_segments)} "
          f"(phi={[segment_phi[s] for s in sorted(unclipped_segments)]})")
    print(f"gated universe: {len(unclipped_months)} months "
          f"(of {sum(len(v) for v in monitored_months_by_segment.values())} total)")
    print()

    import json
    alarms_df = pd.read_csv(ASSETS / "rd_unrate_alarms.csv")
    raw = json.load(open(ASSETS / "exp13_unrate_series_data.json"))
    event_idx = [to_month_index(d) for d in raw["event_months"]]
    hit_window = raw["hit_window"]

    rows = []
    for method in ["raw_cusum", "lsc_kalman_cusum"]:
        alarm_dates = alarms_df[alarms_df.method == method].date.tolist()
        alarm_idx = [to_month_index(pd.Timestamp(d).strftime("%Y-%m-01")) for d in alarm_dates]
        out = gated_permutation_test(alarm_idx, event_idx, unclipped_months,
                                     hit_window, n_perm=n_perm)
        out["method"] = method
        rows.append(out)
        print(f"{method}: gated_hits={out['gated_observed_hits']}/"
              f"{out['gated_n_events']}  gated_alarms={out['gated_n_alarms']}  "
              f"null_mean={out['null_mean']:.3f}  p_gated={out['p_gated']:.4f}")

    pd.DataFrame(rows).to_csv(ASSETS / "exp17_unrate_phi_gated.csv", index=False)
    print(f"\nwrote {ASSETS / 'exp17_unrate_phi_gated.csv'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
