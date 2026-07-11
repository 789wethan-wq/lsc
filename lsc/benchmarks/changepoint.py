"""Benchmark detectors on raw observations.

Online: two-sided Page CUSUM on Y standardized by training-prefix
moments (the classic no-model baseline the diagnostics layer must beat).
Offline: ruptures PELT — retrospective only, never compared on delay
(SPEC §4.1); provided for completeness of the benchmark suite.
"""
from __future__ import annotations

import numpy as np


def raw_cusum_score(Y: np.ndarray, n_train: int, k: float = 0.5) -> np.ndarray:
    """Causal Page-CUSUM score path on raw Y; NaN during the training
    prefix (no online claims on data used for calibration/fitting)."""
    Y = np.asarray(Y, dtype=float)
    mu = Y[:n_train].mean()
    sd = Y[:n_train].std(ddof=1)
    e = (Y - mu) / max(sd, 1e-12)
    T = len(Y)
    out = np.full(T, np.nan)
    gp = gn = 0.0
    for t in range(n_train, T):
        gp = max(0.0, gp + e[t] - k)
        gn = max(0.0, gn - e[t] - k)
        out[t] = max(gp, gn)
    return out


def pelt_breakpoints(Y: np.ndarray, pen: float = 10.0, model: str = "l2") -> list[int]:
    """OFFLINE change-point detection (ruptures PELT). Uses the full
    sample — retrospective analysis only, excluded from delay tables."""
    import ruptures as rpt

    Y = np.asarray(Y, dtype=float).reshape(-1, 1)
    algo = rpt.Pelt(model=model).fit(Y)
    bkps = algo.predict(pen=pen)
    return [b for b in bkps if b < len(Y)]
