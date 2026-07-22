"""Prints the real-data pipeline's actual train/test (monitor) date
boundaries per segment, for every series -- the exact segmentation
real_data.py::run uses (same SERIES config, same load_series, same
T_seg = n_train + n_monitor, same range(0, len(g)-T_seg+1, n_monitor)
loop), so these are read directly off the code and the pinned data
snapshots, not described or hand-computed separately.

Usage: python experiments/real_data_date_boundaries.py [series ...]
       (default: all four series)
Output: prints a table per series; writes
        paper_assets/real_data_date_boundaries.csv (all series combined)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from real_data import SERIES, load_series

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "paper_assets" / "real_data_date_boundaries.csv"


def boundaries_for(series: str) -> pd.DataFrame:
    cfg = dict(SERIES[series])
    g = load_series(cfg, live=False)
    NT, NM = cfg["n_train"], cfg["n_monitor"]
    T_seg = NT + NM

    rows = []
    seg_id = 0
    for start in range(0, len(g) - T_seg + 1, NM):
        seg = g.iloc[start:start + T_seg]
        rows.append(dict(
            series=series, segment=seg_id,
            train_start=f"{seg.index[0]:%Y-%m}",
            train_end=f"{seg.index[NT - 1]:%Y-%m}",
            test_start=f"{seg.index[NT]:%Y-%m}",
            test_end=f"{seg.index[T_seg - 1]:%Y-%m}",
            n_train=NT, n_monitor=NM))
        seg_id += 1
    return pd.DataFrame(rows)


def main(series_list: list[str]) -> None:
    all_rows = []
    for series in series_list:
        df = boundaries_for(series)
        all_rows.append(df)
        print(f"\n=== {series} ({len(df)} segments, "
              f"n_train={df.n_train.iloc[0]}, n_monitor={df.n_monitor.iloc[0]}) ===")
        print(df.to_string(index=False))

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    series_list = sys.argv[1:] if len(sys.argv) > 1 else list(SERIES)
    main(series_list)
