"""Parallelization helper for exp11_break_magnitude_sweep.py: runs a
single (channel, SNR) cell to its own CSV, so the 6 independent cells
(2 channels x 3 SNRs) can be launched as concurrent processes. Purely
a parallelization wrapper -- no change to seeds, n_reps, or any
detector/calibration logic versus running the full sweep serially.

Usage: python experiments/exp11_run_cell.py {level,r} {0.1,0.5,2.0} [n_reps]
Output: paper_assets/exp11_{level,r}channel_snr{snr}.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

from exp11_break_magnitude_sweep import level_sweep, rchannel_sweep

# resolve paper_assets/ relative to the repo root (this file's parent's
# parent), so output location doesn't depend on the launching process's
# working directory -- the bug that lost the first parallel run's results.
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "paper_assets"

channel = sys.argv[1]
snr = float(sys.argv[2])
n_reps = int(sys.argv[3]) if len(sys.argv) > 3 else 500

if channel == "level":
    df = level_sweep(n_reps, snrs=(snr,))
    out = OUT_DIR / f"exp11_levelchannel_snr{snr}.csv"
else:
    df = rchannel_sweep(n_reps, snrs=(snr,))
    out = OUT_DIR / f"exp11_rchannel_snr{snr}.csv"

df.to_csv(out, index=False)
print(f"wrote {out}")
