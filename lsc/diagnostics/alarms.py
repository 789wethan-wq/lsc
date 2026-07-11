"""Alarm scores and null-DGP threshold calibration (SPEC §4.2).

A *detector* is any callable Y -> score path (causal, NaN during
warmup). Calibration runs the detector on ``n_reps`` fresh draws from
the matched null DGP and sets the threshold at the (1 - far) quantile of
the per-replication maximum score, giving P(any alarm in T obs | null)
≈ far. The same routine is used for LSC detectors and benchmarks
(SPEC §9 parity harness).

The composite LSC alarm standardizes each diagnostic feature by its
null interquartile scale (estimated during calibration) and takes the
max |z| across features — then that composite is itself calibrated on
nulls, so multiple-feature testing is automatically accounted for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from lsc.dgp.base import DGP

ScoreFn = Callable[[np.ndarray], np.ndarray]  # Y -> causal score path


@dataclass
class CalibratedDetector:
    name: str
    score_fn: ScoreFn
    threshold: float
    null_max_scores: np.ndarray  # calibration distribution, for diagnostics

    def alarm_time(self, Y: np.ndarray) -> int | None:
        """First index where score >= threshold, else None."""
        score = self.score_fn(np.asarray(Y, dtype=float))
        hits = np.nonzero(np.isfinite(score) & (score >= self.threshold))[0]
        return int(hits[0]) if len(hits) else None

    def alarm_times(self, Y: np.ndarray, rearm_frac: float = 0.5,
                    refractory: int = 20) -> list[int]:
        """All alarms under a re-arm protocol (multi-break evaluation).

        After an alarm the detector is disarmed; it re-arms once the
        score has drained below ``rearm_frac * threshold`` AND at least
        ``refractory`` observations have passed since the alarm. The
        rule is applied identically to every method, so methods whose
        statistics saturate and never drain (e.g. a fixed-baseline
        CUSUM after an unreverted level shift) genuinely cannot signal
        a second event — that is a property of the method, not of the
        protocol. The first element always equals ``alarm_time(Y)``.
        """
        score = self.score_fn(np.asarray(Y, dtype=float))
        alarms: list[int] = []
        armed = True
        for t, s in enumerate(score):
            if not np.isfinite(s):
                continue
            if armed:
                if s >= self.threshold:
                    alarms.append(t)
                    armed = False
            elif (s < rearm_frac * self.threshold
                  and t - alarms[-1] >= refractory):
                armed = True
        return alarms


def calibrate(name: str, score_fn: ScoreFn, null_dgp: DGP, T: int,
              n_reps: int, far: float, seed0: int) -> CalibratedDetector:
    """Set the alarm threshold to achieve FAR ≈ ``far`` per T obs on the
    matched null DGP. Seeds are ``seed0 + i`` — callers must keep
    calibration seeds disjoint from evaluation seeds."""
    max_scores = np.empty(n_reps)
    for i in range(n_reps):
        Y = null_dgp.sample(T, seed=seed0 + i).Y
        score = score_fn(Y)
        finite = score[np.isfinite(score)]
        max_scores[i] = finite.max() if len(finite) else -np.inf
    threshold = float(np.quantile(max_scores, 1.0 - far))
    return CalibratedDetector(name=name, score_fn=score_fn,
                              threshold=threshold, null_max_scores=max_scores)


def empirical_far(det: CalibratedDetector, null_dgp: DGP, T: int,
                  n_reps: int, seed0: int) -> float:
    """Fraction of fresh null draws that trigger any alarm."""
    alarms = 0
    for i in range(n_reps):
        Y = null_dgp.sample(T, seed=seed0 + i).Y
        if det.alarm_time(Y) is not None:
            alarms += 1
    return alarms / n_reps


# ---------------------------------------------------------------------------
# Composite feature score (the LSC diagnostics-layer detector)
# ---------------------------------------------------------------------------

def _robust_scale(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return 1.0
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    return float(iqr / 1.349) if iqr > 1e-12 else max(float(x.std()), 1e-12)


def estimate_feature_scales(feature_paths: list[dict[str, np.ndarray]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-feature, PER-TIME-POINT (center_t, scale_t) across null runs.

    CUSUM-type features grow within a path (nonstationary), so pooling
    their scale over time lets late-time null values dominate and blunts
    the composite for every feature. Standardizing at each t against the
    cross-replication null distribution at that same t makes all
    features comparable at every time point (self-normalization).
    Scales are floored at 10% of the feature's global scale to keep
    early-time z-values from exploding where the null IQR is tiny.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    names = feature_paths[0].keys()
    for name in names:
        stacked = np.stack([fp[name] for fp in feature_paths])  # (R, T)
        with np.errstate(all="ignore"):
            center = np.nanmedian(stacked, axis=0)
            q75 = np.nanpercentile(stacked, 75, axis=0)
            q25 = np.nanpercentile(stacked, 25, axis=0)
        scale = (q75 - q25) / 1.349
        pooled = stacked[np.isfinite(stacked)]
        global_scale = _robust_scale(pooled) if len(pooled) else 1.0
        floor = max(0.1 * global_scale, 1e-12)
        scale = np.where(np.isfinite(scale) & (scale > floor), scale, floor)
        center = np.where(np.isfinite(center), center, 0.0)
        out[name] = (center, scale)
    return out


def composite_score(features: dict[str, np.ndarray],
                    scales: dict[str, tuple[np.ndarray, np.ndarray]],
                    include: list[str] | None = None) -> np.ndarray:
    """max_j |feature_j(t) - center_j(t)| / scale_j(t); NaN where all
    features are NaN."""
    names = include if include is not None else list(features.keys())
    T = len(next(iter(features.values())))
    z = np.full((len(names), T), np.nan)
    for j, name in enumerate(names):
        center, scale = scales[name]
        z[j] = np.abs(features[name] - center) / scale
    all_nan = np.all(~np.isfinite(z), axis=0)
    z[:, all_nan] = -np.inf  # avoid nanmax All-NaN warning
    score = np.nanmax(z, axis=0)
    score[all_nan] = np.nan
    return score
