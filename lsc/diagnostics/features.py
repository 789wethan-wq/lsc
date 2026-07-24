"""Diagnostics features from the *filtered* latent path (SPEC §5, M3).

Every feature at time t is a function of {filtered[0..t], innovations[0..t]}
only — rolling windows look strictly backward. Warmup positions (fewer
than ``window`` past observations) are NaN and must be excluded from
alarm scoring by masking, not by imputation.

Features:
  level_change   filtered[t] - filtered[t-w]
  slope          OLS slope of filtered over the past w obs
  acceleration   slope[t] - slope[t-w]
  instability    rolling std of filtered increments over past w
  persistence    rolling AR(1) coefficient of filtered increments
  break_pressure two-sided Page CUSUM of standardized innovations
"""
from __future__ import annotations

import numpy as np

from lsc.models.base import StateEstimate


def _rolling_apply(x: np.ndarray, w: int, fn) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for t in range(w, len(x)):
        out[t] = fn(x[t - w + 1: t + 1])
    return out


def level_change(filtered: np.ndarray, window: int = 20) -> np.ndarray:
    out = np.full(len(filtered), np.nan)
    out[window:] = filtered[window:] - filtered[:-window]
    return out


def slope(filtered: np.ndarray, window: int = 20) -> np.ndarray:
    t = np.arange(window, dtype=float)
    t -= t.mean()
    denom = (t**2).sum()
    out = np.full(len(filtered), np.nan)
    for i in range(window - 1, len(filtered)):
        seg = filtered[i - window + 1: i + 1]
        out[i] = (t * (seg - seg.mean())).sum() / denom
    return out


def acceleration(filtered: np.ndarray, window: int = 20) -> np.ndarray:
    sl = slope(filtered, window)
    out = np.full(len(filtered), np.nan)
    out[window:] = sl[window:] - sl[:-window]
    return out


def instability(filtered: np.ndarray, window: int = 20) -> np.ndarray:
    inc = np.diff(filtered, prepend=filtered[0])
    return _rolling_apply(inc, window, lambda seg: seg.std(ddof=1))


def persistence(filtered: np.ndarray, window: int = 40) -> np.ndarray:
    inc = np.diff(filtered, prepend=filtered[0])

    def ar1(seg: np.ndarray) -> float:
        a, b = seg[:-1] - seg[:-1].mean(), seg[1:] - seg[1:].mean()
        den = (a**2).sum()
        return float((a * b).sum() / den) if den > 1e-12 else 0.0

    return _rolling_apply(inc, window, ar1)


def break_pressure(innovations: np.ndarray, k: float = 0.5,
                   warmup: int = 10) -> np.ndarray:
    """Two-sided Page CUSUM of standardized innovations.

    g+_t = max(0, g+_{t-1} + e_t - k);  g-_t analogous with -e_t.
    Returns max(g+, g-); first ``warmup`` values NaN (diffuse filter
    initialization makes early innovations unreliable).
    """
    T = len(innovations)
    out = np.full(T, np.nan)
    gp = gn = 0.0
    for t in range(T):
        e = innovations[t]
        if not np.isfinite(e):
            e = 0.0
        gp = max(0.0, gp + e - k)
        gn = max(0.0, gn - e - k)
        if t >= warmup:
            out[t] = max(gp, gn)
    return out


def windowed_break_pressure(innovations: np.ndarray, window: int = 60,
                            warmup: int = 10) -> np.ndarray:
    """MOSUM-style bounded-memory alternative to break_pressure: a
    moving two-window mean-shift statistic (Chu, Stinchcombe & White
    1996 fluctuation-test family) rather than a CUSUM accumulated from
    the training baseline.

    At each t, compares the mean of the trailing ``window``
    innovations ("test") against the mean of the ``window``
    innovations immediately before that ("reference") via a
    two-sample z-statistic. A first attempt restarted a Page CUSUM
    accumulator every ``window`` obs but STILL compared against the
    fixed reference k/0 baseline, so it reproduced the same
    steady-state elevated value in every window after a permanent
    shift and never drained (caught by
    tests/test_multibreak.py::test_windowed_break_pressure_drains_
    after_permanent_shift, CHANGELOG P2 2026-07-16) — the reference
    itself has to be local and sliding, not fixed, for the statistic
    to forget an old, already-absorbed shift. Here both windows slide
    forward together, so ``window`` obs after a permanent shift both
    windows sit inside the same new regime and the statistic returns
    to its null level, while a shift *between* the two windows still
    produces a sharp, transient peak. O(T*window)."""
    T = len(innovations)
    e = np.where(np.isfinite(innovations), innovations, 0.0)
    out = np.full(T, np.nan)
    for t in range(warmup, T):
        if t - 2 * window + 1 < 0:
            continue
        test = e[t - window + 1: t + 1]
        ref = e[t - 2 * window + 1: t - window + 1]
        se = np.sqrt(test.var(ddof=1) / window + ref.var(ddof=1) / window)
        out[t] = abs(test.mean() - ref.mean()) / max(se, 1e-6)
    return out


def variance_pressure(innovations: np.ndarray, k: float = 0.25,
                      warmup: int = 10) -> np.ndarray:
    """One-sided CUSUM of (e_t^2 - 1): pressure toward variance increase."""
    T = len(innovations)
    out = np.full(T, np.nan)
    g = 0.0
    for t in range(T):
        e = innovations[t]
        if not np.isfinite(e):
            e = 0.0
        g = max(0.0, g + (e * e - 1.0) - k)
        if t >= warmup:
            out[t] = g
    return out


def variance_quiet(innovations: np.ndarray, k: float = 0.05,
                   warmup: int = 10) -> np.ndarray:
    """One-sided CUSUM of (1 - e_t^2): pressure toward variance DECREASE
    ("the series got quieter"). exp03 showed quieting dynamics suppress
    every excursion-based statistic — this is the dedicated statistic."""
    T = len(innovations)
    out = np.full(T, np.nan)
    g = 0.0
    for t in range(T):
        e = innovations[t]
        if not np.isfinite(e):
            e = 0.0
        g = max(0.0, g + (1.0 - e * e) - k)
        if t >= warmup:
            out[t] = g
    return out


def _robust_variance_summand(innovations: np.ndarray, n_train: int,
                             clip: float = 2.5,
                             warmup: int = 10) -> np.ndarray:
    """Standardized, CLIPPED second-moment summand (exp05).

    Innovations are rescaled by their training-prefix robust scale
    (1.4826*MAD), clipped at ±clip, squared, then centered and
    standardized by the training-prefix mean/sd of the clipped square.
    All reference moments come from [warmup, n_train) only (causal).
    Clipping bounds each summand by (clip² − m̂)/ŝ, so the null CUSUM
    max is thin-tailed regardless of the innovation distribution —
    the repair for the t₅ collapse seen in grid_v2_misspec.
    """
    e = np.where(np.isfinite(innovations), innovations, 0.0)
    train = e[warmup:n_train]
    mad = np.median(np.abs(train - np.median(train)))
    scale = max(1.4826 * mad, 1e-12)
    sq = np.clip(e / scale, -clip, clip) ** 2
    m = sq[warmup:n_train].mean()
    s = max(sq[warmup:n_train].std(ddof=1), 1e-12)
    return (sq - m) / s


def variance_pressure_robust(innovations: np.ndarray, n_train: int,
                             k: float = 0.05, clip: float = 2.5,
                             warmup: int = 10) -> np.ndarray:
    """One-sided CUSUM of the clipped standardized second moment:
    tail-robust pressure toward variance increase."""
    x = _robust_variance_summand(innovations, n_train, clip, warmup)
    T = len(x)
    out = np.full(T, np.nan)
    g = 0.0
    for t in range(T):
        g = max(0.0, g + x[t] - k)
        if t >= warmup:
            out[t] = g
    return out


def variance_quiet_robust(innovations: np.ndarray, n_train: int,
                          k: float = 0.05, clip: float = 2.5,
                          warmup: int = 10) -> np.ndarray:
    """Tail-robust pressure toward variance DECREASE (quieting)."""
    x = _robust_variance_summand(innovations, n_train, clip, warmup)
    T = len(x)
    out = np.full(T, np.nan)
    g = 0.0
    for t in range(T):
        g = max(0.0, g - x[t] - k)
        if t >= warmup:
            out[t] = g
    return out


def _exceedance_series(innovations: np.ndarray, n_train: int,
                       q: float = 0.90,
                       warmup: int = 10) -> tuple[np.ndarray, float]:
    """Indicator of |e_t| exceeding its training-prefix q-quantile,
    plus the training exceedance rate p̂ (≈ 1−q by construction).

    The exp05b repair for heavy tails: the summand is BOUNDED under any
    innovation distribution (thin-tailed null CUSUM maxima), while a
    scale rise moves the exceedance probability strongly — tail
    sensitivity through the indicator, not the magnitude (clipping the
    magnitude, exp05, destroyed the signal along with the null tail).
    """
    e = np.where(np.isfinite(innovations), innovations, 0.0)
    a = np.abs(e)
    thr = np.quantile(a[warmup:n_train], q)
    ind = (a > thr).astype(float)
    p_hat = float(ind[warmup:n_train].mean())
    return ind, p_hat


def tail_exceedance(innovations: np.ndarray, n_train: int, k: float = 0.05,
                    q: float = 0.90, warmup: int = 10) -> np.ndarray:
    """One-sided CUSUM of (1{|e|>q̂} − p̂ − k): tail-robust pressure
    toward variance increase. k=0.05 chosen on non-evaluation seeds
    (exp05c CHANGELOG entry)."""
    ind, p_hat = _exceedance_series(innovations, n_train, q, warmup)
    out = np.full(len(ind), np.nan)
    g = 0.0
    for t in range(len(ind)):
        g = max(0.0, g + ind[t] - p_hat - k)
        if t >= warmup:
            out[t] = g
    return out


def tail_shortfall(innovations: np.ndarray, n_train: int, k: float = 0.02,
                   q: float = 0.90, warmup: int = 10) -> np.ndarray:
    """One-sided CUSUM of (p̂ − 1{|e|>q̂} − k): tail-robust pressure
    toward variance DECREASE (quieting)."""
    ind, p_hat = _exceedance_series(innovations, n_train, q, warmup)
    out = np.full(len(ind), np.nan)
    g = 0.0
    for t in range(len(ind)):
        g = max(0.0, g + p_hat - ind[t] - k)
        if t >= warmup:
            out[t] = g
    return out


def innovation_ac(innovations: np.ndarray, window: int = 40) -> np.ndarray:
    """Rolling lag-1 autocorrelation of innovations. Under a correctly
    specified filter innovations are white; a change in the state's
    dynamics makes the fixed-parameter filter misspecified and its
    innovations serially correlated — in either direction, so this is
    consumed two-sided by the composite."""
    e = np.where(np.isfinite(innovations), innovations, 0.0)

    def ac1(seg: np.ndarray) -> float:
        a, b = seg[:-1] - seg[:-1].mean(), seg[1:] - seg[1:].mean()
        den = np.sqrt((a**2).sum() * (b**2).sum())
        return float((a * b).sum() / den) if den > 1e-12 else 0.0

    return _rolling_apply(e, window, ac1)


def state_shift_pressure(filtered: np.ndarray, n_train: int, k: float = 0.5,
                         warmup: int = 20) -> np.ndarray:
    """Two-sided Page CUSUM of the filtered state standardized by its
    training-prefix moments — the latent-space analog of the raw-Y
    baseline CUSUM. An adaptive filter tracks a persistent state shift,
    so post-break the standardized filtered state stays displaced from
    the training baseline and the statistic accumulates the FULL shift
    per step (innovations only carry (1-phi) of it), with observation
    noise already filtered out."""
    base = filtered[warmup:n_train]
    mu = base.mean()
    sd = max(base.std(ddof=1), 1e-12)
    z = (filtered - mu) / sd
    T = len(filtered)
    out = np.full(T, np.nan)
    gp = gn = 0.0
    for t in range(warmup, T):
        gp = max(0.0, gp + z[t] - k)
        gn = max(0.0, gn - z[t] - k)
        out[t] = max(gp, gn)
    return out


FEATURE_FNS = {
    "level_change": lambda est, w, nt: level_change(est.filtered, w),
    "slope": lambda est, w, nt: slope(est.filtered, w),
    "acceleration": lambda est, w, nt: acceleration(est.filtered, w),
    "instability": lambda est, w, nt: instability(est.filtered, w),
    "persistence": lambda est, w, nt: persistence(est.filtered, max(w, 40)),
    "break_pressure": lambda est, w, nt: break_pressure(est.innovations),
    "variance_pressure": lambda est, w, nt: variance_pressure(est.innovations),
    "variance_pressure_slow": lambda est, w, nt: variance_pressure(est.innovations, k=0.05),
    "variance_quiet": lambda est, w, nt: variance_quiet(est.innovations),
    "innovation_ac": lambda est, w, nt: innovation_ac(est.innovations),
    "state_shift_pressure": lambda est, w, nt: state_shift_pressure(est.filtered, nt),
    "variance_pressure_robust": lambda est, w, nt: variance_pressure_robust(est.innovations, nt),
    "variance_quiet_robust": lambda est, w, nt: variance_quiet_robust(est.innovations, nt),
    "tail_exceedance": lambda est, w, nt: tail_exceedance(est.innovations, nt),
    "tail_shortfall": lambda est, w, nt: tail_shortfall(est.innovations, nt),
}

# Frozen include-lists for the composite detector. COMPOSITE_V1 is the
# 11-feature set used by every experiment through grid_v2/exp04 —
# freezing it keeps those results bit-reproducible as FEATURE_FNS
# grows. COMPOSITE_ROBUST (exp05) swaps the three e²-based variance
# features for the two clipped/tail-robust ones.
COMPOSITE_V1 = [
    "level_change", "slope", "acceleration", "instability", "persistence",
    "break_pressure", "variance_pressure", "variance_pressure_slow",
    "variance_quiet", "innovation_ac", "state_shift_pressure",
]
COMPOSITE_ROBUST = [
    "level_change", "slope", "acceleration", "instability", "persistence",
    "break_pressure", "variance_pressure_robust", "variance_quiet_robust",
    "innovation_ac", "state_shift_pressure",
]
# exp05b: exceedance-indicator variance features (exp05's clip-based
# set is a documented negative result — kept above, recommended nowhere)
COMPOSITE_ROBUST2 = [
    "level_change", "slope", "acceleration", "instability", "persistence",
    "break_pressure", "tail_exceedance", "tail_shortfall",
    "innovation_ac", "state_shift_pressure",
]
# exp21: the 5/11 COMPOSITE_V1 features that act on `innovations` alone
# (no filtered-state dependence) — exp20's disclosed DIRECT-substitution
# subset, isolated here to test whether the Kalman-vs-ARIMA composite gap
# already opens on the innovation series alone or only once the six
# filtered-state features (level_change, slope, acceleration,
# instability, persistence, state_shift_pressure) are added back.
COMPOSITE_INNOV5 = [
    "break_pressure", "variance_pressure", "variance_pressure_slow",
    "variance_quiet", "innovation_ac",
]


def compute_features(est: StateEstimate, window: int = 20,
                     n_train: int | None = None) -> dict[str, np.ndarray]:
    """All diagnostic features. ``n_train`` is required by baseline-
    referenced features; if None, those are omitted."""
    fns = dict(FEATURE_FNS)
    if n_train is None:
        for name in ("state_shift_pressure", "variance_pressure_robust",
                     "variance_quiet_robust", "tail_exceedance",
                     "tail_shortfall"):
            fns.pop(name)
        n_train = 0
    return {name: fn(est, window, n_train) for name, fn in fns.items()}
