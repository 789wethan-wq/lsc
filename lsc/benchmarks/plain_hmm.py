"""Plain-HMM regime-flip benchmark: alarm when the causal filtered
probability of leaving the training-dominant regime gets large, with no
diagnostics layer on top."""
from __future__ import annotations

import numpy as np

from lsc.models.hmm import HMMModel


def plain_hmm_flip_score(Y: np.ndarray, n_train: int, n_regimes: int = 2,
                         em_seed: int = 0) -> np.ndarray:
    Y = np.asarray(Y, dtype=float)
    model = HMMModel(n_regimes, em_seed=em_seed).fit(Y[:n_train])
    est = model.filter(Y)
    probs = est.filtered_probs  # (T, K), causal
    # dominant regime over the training prefix
    dom = int(np.argmax(probs[:n_train].mean(axis=0)))
    # log-odds of having left the dominant regime; unbounded, so a 5%-FAR
    # threshold always exists (raw probability saturates at 1.0)
    p = np.clip(probs[:, dom], 1e-12, 1 - 1e-12)
    score = np.log1p(-p) - np.log(p)
    score[:n_train] = np.nan
    return score
