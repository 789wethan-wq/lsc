"""Detection and state-recovery metrics (SPEC §7)."""
from __future__ import annotations

import numpy as np


def detection_outcome(alarm_time: int | None, break_time: int, T: int) -> dict:
    """Classify one replication's alarm against a single true break.

    - false_alarm: alarm strictly before the break
    - detected:    alarm at/after the break
    - missed:      no alarm by T (delay censored at T - break_time)
    """
    if alarm_time is None:
        return dict(detected=False, false_alarm=False, missed=True,
                    delay=float(T - break_time), delay_censored=True)
    if alarm_time < break_time:
        return dict(detected=False, false_alarm=True, missed=False,
                    delay=np.nan, delay_censored=False)
    return dict(detected=True, false_alarm=False, missed=False,
                delay=float(alarm_time - break_time), delay_censored=False)


def summarize_detection(outcomes: list[dict]) -> dict:
    """MC summary with Monte Carlo standard errors (SPEC §4.4)."""
    n = len(outcomes)
    det = np.array([o["detected"] for o in outcomes], dtype=float)
    fa = np.array([o["false_alarm"] for o in outcomes], dtype=float)
    miss = np.array([o["missed"] for o in outcomes], dtype=float)
    # censored delays included at censoring value: honest upper-bound summary
    delays = np.array([o["delay"] for o in outcomes], dtype=float)
    delays_cens = delays[~np.isnan(delays)]
    det_delays = delays[det.astype(bool)]

    def se(x):
        return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan

    return dict(
        n=n,
        detect_rate=float(det.mean()), detect_rate_se=se(det),
        pre_break_false_alarm_rate=float(fa.mean()), pre_break_false_alarm_rate_se=se(fa),
        miss_rate=float(miss.mean()), miss_rate_se=se(miss),
        mean_delay_censored=float(delays_cens.mean()), mean_delay_censored_se=se(delays_cens),
        median_delay_detected=float(np.median(det_delays)) if len(det_delays) else np.nan,
        mean_delay_detected=float(det_delays.mean()) if len(det_delays) else np.nan,
        mean_delay_detected_se=se(det_delays),
    )


def multi_break_outcome(alarm_times: list[int], break_times: list[int],
                        T: int, window: int = 100) -> dict:
    """Event-level matching for multi-break paths (SPEC §7).

    Breaks (in time order) are matched greedily one-to-one to the first
    unused alarm in [break, break + window). Unmatched alarms are false
    positives (this includes any alarm before the first break);
    unmatched breaks are misses. F1 = 2*TP / (2*TP + FP + FN) is
    well-defined even with zero alarms; precision is NaN when there are
    no alarms (excluded from its MC average, reported via n_alarms).
    """
    alarms = sorted(int(a) for a in alarm_times)
    used = [False] * len(alarms)
    tp, delays, matched = 0, [], []
    for b in sorted(break_times):
        hit = False
        for i, a in enumerate(alarms):
            if not used[i] and b <= a < b + window:
                used[i] = True
                tp += 1
                delays.append(float(a - b))
                hit = True
                break
        matched.append(hit)
    fp = used.count(False)
    fn = len(break_times) - tp
    return dict(
        n_alarms=len(alarms), tp=tp, fp=fp, fn=fn, matched=matched,
        precision=(tp / len(alarms) if alarms else np.nan),
        recall=tp / len(break_times),
        f1=2.0 * tp / (2.0 * tp + fp + fn) if (tp + fp + fn) else 1.0,
        mean_matched_delay=(float(np.mean(delays)) if delays else np.nan),
    )


def summarize_multi_break(outcomes: list[dict]) -> dict:
    """MC summary of multi-break outcomes with MC standard errors."""
    def col(key):
        return np.array([o[key] for o in outcomes], dtype=float)

    def mean_se(x):
        x = x[np.isfinite(x)]
        if not len(x):
            return np.nan, np.nan
        se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan
        return float(x.mean()), se

    out = dict(n=len(outcomes))
    for key in ("precision", "recall", "f1", "n_alarms",
                "mean_matched_delay", "fp"):
        m, se = mean_se(col(key))
        out[key], out[f"{key}_se"] = m, se
    return out


def state_recovery(filtered: np.ndarray, S_true: np.ndarray,
                   start: int = 0) -> dict:
    from scipy.stats import pearsonr, spearmanr

    f, s = filtered[start:], S_true[start:]
    return dict(
        pearson=float(pearsonr(f, s)[0]),
        spearman=float(spearmanr(f, s)[0]),
        rmse=float(np.sqrt(np.mean((f - s) ** 2))),
    )
