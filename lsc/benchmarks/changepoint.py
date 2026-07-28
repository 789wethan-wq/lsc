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


def windowed_raw_cusum_score(Y: np.ndarray, n_train: int,
                             window: int = 60) -> np.ndarray:
    """Bounded-memory, MOSUM-style counterpart of raw_cusum_score
    (Chu, Hornik & Kuan 1995 MOSUM family): a moving
    two-window mean-shift z-statistic on standardized Y, rather than a
    CUSUM accumulated against the training-prefix baseline. See
    windowed_break_pressure for the full rationale (a naive
    restart-every-window CUSUM against a FIXED reference does not
    drain after a permanent shift; this two-window design does,
    because both windows slide and eventually sit inside the same new
    regime). Fixes the second-event miss documented for raw_cusum in
    exp04 (CHANGELOG P2 2026-07-16)."""
    Y = np.asarray(Y, dtype=float)
    mu = Y[:n_train].mean()
    sd = Y[:n_train].std(ddof=1)
    e = (Y - mu) / max(sd, 1e-12)
    T = len(Y)
    out = np.full(T, np.nan)
    for t in range(n_train, T):
        if t - n_train - 2 * window + 1 < 0:
            continue
        test = e[t - window + 1: t + 1]
        ref = e[t - 2 * window + 1: t - window + 1]
        se = np.sqrt(test.var(ddof=1) / window + ref.var(ddof=1) / window)
        out[t] = abs(test.mean() - ref.mean()) / max(se, 1e-6)
    return out


def pelt_breakpoints(Y: np.ndarray, pen: float = 10.0, model: str = "l2") -> list[int]:
    """OFFLINE change-point detection (ruptures PELT). Uses the full
    sample — retrospective analysis only, excluded from delay tables."""
    import ruptures as rpt

    Y = np.asarray(Y, dtype=float).reshape(-1, 1)
    algo = rpt.Pelt(model=model).fit(Y)
    bkps = algo.predict(pen=pen)
    return [b for b in bkps if b < len(Y)]


def _icss_dmax(seg: np.ndarray) -> tuple[int, float]:
    """Single-window Inclan-Tiao (1994) D-statistic: D_k = C_k/C_T -
    k/T, C_k = cumsum(seg**2)[k]; returns (k_star, |D_k*|) for the
    maximizing k (0-indexed within seg — seg[:k_star+1] is "before")."""
    n = len(seg)
    if n < 4:
        return -1, 0.0
    csq = np.cumsum(np.asarray(seg, dtype=float) ** 2)
    CT = csq[-1]
    if CT <= 0:
        return -1, 0.0
    k = np.arange(1, n)
    Dk = csq[:-1] / CT - k / n
    idx = int(np.argmax(np.abs(Dk)))
    return idx, float(np.abs(Dk[idx]))


def icss_breakpoints(Y: np.ndarray, crit: float, min_seg: int = 8) -> list[int]:
    """OFFLINE variance-change-point detection (Inclan & Tiao 1994,
    "Use of Cumulative Sums of Squares for Retrospective Detection of
    Changes of Variance"). Recursively partitions the series at the
    point of maximal normalized cumulative-sum-of-squares deviation,
    accepting a candidate breakpoint whenever its D-statistic exceeds
    `crit` and recursing on both halves — the standard ICSS iterative
    search for possibly-multiple variance breaks. `crit` is calibrated
    by simulation to a target false-alarm rate (see
    experiments/exp25_icss_benchmark.py), the same calibrated-parity
    convention this repo uses for PELT's `pen`, rather than
    Inclan-Tiao's asymptotic critical value — the raw, scale-free D_k
    statistic (range roughly [-1, 1]) does not need that rescaling
    once the threshold itself is set empirically."""
    Y = np.asarray(Y, dtype=float)
    breakpoints: list[int] = []

    def recurse(start: int, end: int) -> None:
        seg = Y[start:end]
        if len(seg) < min_seg:
            return
        k_star, d_max = _icss_dmax(seg)
        if k_star < 0 or d_max <= crit:
            return
        bp = start + k_star + 1  # first index of the "after" segment
        breakpoints.append(bp)
        recurse(start, bp)
        recurse(bp, end)

    recurse(0, len(Y))
    return sorted(breakpoints)
