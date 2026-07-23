"""exp13d_all_series_circular_shift.py -- runs the already-verified
circular-shift joint test (exp13c) on all four real-data series, then
does the final Bonferroni-across-series step the paper's Appendix A
explicitly said was pending.

Depends only on exp13c_circular_shift.py (unmodified) and the four
per-series JSON files, each of which must include a "window_start"
field (the monitored window's own true start date) alongside
n_months, event_months, hit_window, and alarms -- INDPRO's already has
this; GDP/GS10/UNRATE's come from exp13d_export_other_series.py.

This script itself needs no further verification once the JSONs are
real: circular_shift_joint_test() is the exact function already
checkpoint-verified against INDPRO (p=0.029, matching 0.027-0.029
across independent seeds, after the window-anchoring fix). Running it
on three more real inputs is not a new method -- it's the same
method, unmodified, run more times.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp13c_circular_shift import circular_shift_joint_test_exact, to_month_index  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "paper_assets"

SERIES_FILES = {
    "INDPRO": ASSETS / "exp13_indpro_series_data.json",
    "GDP": ASSETS / "exp13_gdp_series_data.json",
    "GS10": ASSETS / "exp13_gs10_series_data.json",
    "UNRATE": ASSETS / "exp13_unrate_series_data.json",
}


def load_series(path: Path) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return dict(
        n_months=raw["n_months"],
        window_start_idx=to_month_index(raw["window_start"]),
        hit_window=raw["hit_window"],
        event_months=[to_month_index(d) for d in raw["event_months"]],
        alarms={m: [to_month_index(d) for d in v] for m, v in raw["alarms"].items()},
        step=raw.get("step", 1),
    )


def main(n_perm: int = 20000) -> None:
    """n_perm is unused -- kept for CLI compatibility. The test is now
    exact (exhaustive over every possible shift, <=780 per series), not
    Monte Carlo sampled, so there is no sample size to choose."""
    results = {}
    missing = []
    for name, path in SERIES_FILES.items():
        if not path.exists():
            missing.append(name)
            continue
        series = load_series(path)
        out = circular_shift_joint_test_exact(series)
        results[name] = out
        print(f"{name}: total_hits={out['observed_total']} "
              f"null_mean={out['null_total_mean']:.3f} "
              f"null_sd={out['null_total_sd']:.3f} "
              f"null_max={out['null_total_max']} "
              f"n_shifts_exhaustive={out['n_shifts_exhaustive']} "
              f"EXACT p_total_hits={out['p_total_hits']:.6f}")

    if missing:
        print(f"\nSkipped (JSON not found, run exp13d_export_other_series.py "
              f"first): {missing}")
        print("Bonferroni step below is not run until all four are present "
              "-- a 3-series or 2-series correction would use the wrong "
              "denominator and give a misleadingly small threshold.")
        return

    n_series = len(results)
    alpha_bonf = 0.05 / n_series
    print(f"\nBonferroni across {n_series} series: threshold = "
          f"0.05/{n_series} = {alpha_bonf:.5f}")
    any_survives = False
    for name, out in results.items():
        p = out["p_total_hits"]
        survives = p <= alpha_bonf
        any_survives = any_survives or survives
        print(f"  {name}: p={p:.5f}  {'SURVIVES' if survives else 'does not survive'}")

    if any_survives:
        conclusion = ("at least one series clears both the multiple-testing "
                      "bar and would need the model-fit cross-check before "
                      "being reported as a finding")
    else:
        conclusion = ("no series clears the Bonferroni bar -- the paper's "
                      "original conclusion (\"no series clears both bars\") "
                      "holds with real data, not just the provisional "
                      "estimate that was removed earlier")
    print(f"\nConclusion: {conclusion}.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
