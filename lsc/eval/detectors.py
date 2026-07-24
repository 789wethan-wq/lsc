"""Detector factories: bundle (model fit-on-prefix → causal filter →
diagnostics → score) into ScoreFn callables that the shared calibration
harness consumes. Benchmarks and LSC detectors are built through the
same machinery so they get identical data splits and calibration
budgets (SPEC §9 parity).

All detectors return NaN scores for t < n_train: no online detection
claims on the data used for parameter estimation.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from lsc.benchmarks.changepoint import raw_cusum_score
from lsc.benchmarks.plain_hmm import plain_hmm_flip_score
from lsc.diagnostics.alarms import (
    ScoreFn,
    composite_score,
    estimate_feature_scales,
)
from lsc.diagnostics.features import (
    COMPOSITE_V1,
    break_pressure,
    compute_features,
)
from lsc.dgp.base import DGP
from lsc.models.base import Model


def _mask_train(score: np.ndarray, n_train: int) -> np.ndarray:
    score = score.copy()
    score[:n_train] = np.nan
    return score


def make_innovation_cusum_detector(model_factory: Callable[[], Model],
                                   n_train: int, k: float = 0.5) -> ScoreFn:
    """LSC break-pressure detector: CUSUM of the model's standardized
    causal innovations (the primary single-feature detector)."""

    def score_fn(Y: np.ndarray) -> np.ndarray:
        est = model_factory().fit_filter(Y, n_train=n_train)
        return _mask_train(break_pressure(est.innovations, k=k), n_train)

    return score_fn


def make_composite_detector(model_factory: Callable[[], Model],
                            null_dgp: DGP, T: int, n_train: int,
                            window: int = 20, n_scale_reps: int = 50,
                            scale_seed0: int = 900_000,
                            include: list[str] | None = None) -> ScoreFn:
    """LSC composite detector over an include-list of diagnostic
    features (default: the frozen 11-feature COMPOSITE_V1 set, so
    results stay reproducible as FEATURE_FNS grows; pass
    features.COMPOSITE_ROBUST for the tail-robust variant).

    Feature centers/scales are estimated once from ``n_scale_reps`` null
    runs (seeds disjoint from both calibration and evaluation seeds),
    then frozen inside the returned ScoreFn.
    """
    if include is None:
        include = COMPOSITE_V1
    paths = []
    for i in range(n_scale_reps):
        Ynull = null_dgp.sample(T, seed=scale_seed0 + i).Y
        est = model_factory().fit_filter(Ynull, n_train=n_train)
        feats = compute_features(est, window=window, n_train=n_train)
        # only post-train segment informs the null scale
        paths.append({k: _mask_train(v, n_train) for k, v in feats.items()})
    scales = estimate_feature_scales(paths)

    def score_fn(Y: np.ndarray) -> np.ndarray:
        est = model_factory().fit_filter(Y, n_train=n_train)
        feats = compute_features(est, window=window, n_train=n_train)
        return _mask_train(composite_score(feats, scales, include), n_train)

    # exposed for alarm attribution (which feature crossed): real_data.py
    score_fn.scales = scales
    score_fn.include = include
    score_fn.window = window
    score_fn.model_factory = model_factory
    score_fn.n_train = n_train
    return score_fn


def make_windowed_innovation_cusum_detector(model_factory: Callable[[], Model],
                                            n_train: int,
                                            window: int = 60) -> ScoreFn:
    """Bounded-memory counterpart of make_innovation_cusum_detector
    (P2 exp04 fix): a MOSUM-style two-window mean-shift statistic on
    the model's causal innovations, so it can drain and re-arm for a
    second break (see windowed_break_pressure)."""
    from lsc.diagnostics.features import windowed_break_pressure

    def score_fn(Y: np.ndarray) -> np.ndarray:
        est = model_factory().fit_filter(Y, n_train=n_train)
        return _mask_train(
            windowed_break_pressure(est.innovations, window=window),
            n_train)

    return score_fn


def make_windowed_raw_cusum_detector(n_train: int, window: int = 60) -> ScoreFn:
    """Bounded-memory counterpart of make_raw_cusum_detector (P2 exp04
    fix)."""
    from lsc.benchmarks.changepoint import windowed_raw_cusum_score

    def score_fn(Y: np.ndarray) -> np.ndarray:
        return windowed_raw_cusum_score(Y, n_train=n_train, window=window)

    return score_fn


def make_state_cusum_detector(model_factory: Callable[[], Model],
                              n_train: int, k: float = 0.5) -> ScoreFn:
    """LSC state-shift detector: baseline CUSUM of the filtered state —
    the direct latent-space counterpart of the raw-Y CUSUM benchmark."""
    from lsc.diagnostics.features import state_shift_pressure

    def score_fn(Y: np.ndarray) -> np.ndarray:
        est = model_factory().fit_filter(Y, n_train=n_train)
        return _mask_train(state_shift_pressure(est.filtered, n_train, k=k),
                           n_train)

    return score_fn


def make_tail_cusum_detector(model_factory: Callable[[], Model],
                             n_train: int) -> ScoreFn:
    """Standalone tail-robust variance detector (exp05c): max of the
    exceedance (scale-up) and shortfall (scale-down) indicator CUSUMs.
    Bounded increments give thin-tailed null maxima under ANY
    innovation distribution."""
    from lsc.diagnostics.features import tail_exceedance, tail_shortfall

    def score_fn(Y: np.ndarray) -> np.ndarray:
        est = model_factory().fit_filter(Y, n_train=n_train)
        up = tail_exceedance(est.innovations, n_train)
        down = tail_shortfall(est.innovations, n_train)
        return _mask_train(np.fmax(up, down), n_train)

    return score_fn


def make_arima_cusum_detector(n_train: int, k: float = 0.5) -> ScoreFn:
    from lsc.benchmarks.arima import arima_cusum_score

    def score_fn(Y: np.ndarray) -> np.ndarray:
        return arima_cusum_score(Y, n_train=n_train, k=k)

    return score_fn


def make_raw_cusum_detector(n_train: int, k: float = 0.5) -> ScoreFn:
    def score_fn(Y: np.ndarray) -> np.ndarray:
        return raw_cusum_score(Y, n_train=n_train, k=k)

    return score_fn


def make_raw_var_cusum_detector(n_train: int) -> ScoreFn:
    """Whitening-ladder bottom rung: variance CUSUM on raw Y
    standardized by frozen training-prefix moments (SPEC addendum §2)."""
    from lsc.benchmarks.variance import raw_var_cusum_score

    def score_fn(Y: np.ndarray) -> np.ndarray:
        return raw_var_cusum_score(Y, n_train=n_train)

    return score_fn


def make_windowed_raw_var_cusum_detector(n_train: int, window: int = 60) -> ScoreFn:
    """Bounded-memory, MOSUM-style counterpart of
    make_raw_var_cusum_detector (peer review round 3, Missing
    Experiments) — the variance-channel analog of
    make_windowed_raw_cusum_detector, which fixes second-event misses
    for MEAN shifts but not variance ones (exp04's var_up_down: 0.00
    recall on the second event)."""
    from lsc.benchmarks.variance import windowed_raw_var_score

    def score_fn(Y: np.ndarray) -> np.ndarray:
        return windowed_raw_var_score(Y, n_train=n_train, window=window)

    return score_fn


def make_arima_var_cusum_detector(n_train: int) -> ScoreFn:
    """Whitening-ladder middle rung: the identical variance statistic
    on the frozen training-prefix ARIMA model's residuals."""
    from lsc.benchmarks.variance import arima_var_cusum_score

    def score_fn(Y: np.ndarray) -> np.ndarray:
        return arima_var_cusum_score(Y, n_train=n_train)

    return score_fn


def make_plain_hmm_detector(n_train: int, n_regimes: int = 2) -> ScoreFn:
    def score_fn(Y: np.ndarray) -> np.ndarray:
        return plain_hmm_flip_score(Y, n_train=n_train, n_regimes=n_regimes)

    return score_fn
