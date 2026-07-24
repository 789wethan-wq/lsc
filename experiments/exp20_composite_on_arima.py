"""exp20_composite_on_arima.py -- composite-on-ARIMA ablation: does the
LSC composite's power over plain ARIMA-CUSUM come only from richer
statistics on the SAME whitened innovation series, or partly from
something genuinely state-specific?

The paper's "the ladder is really raw vs. whitened" claim (Sec 5) rests
on the ARMA(1,1) equivalence (`lsc.theory.arma11_representation`,
`experiments/exp07_arma_equivalence.py`) -- but that equivalence is
proven for the INNOVATION SERIES, not for the full 11-feature
composite. Several composite features (level_change, slope,
acceleration, instability, persistence, state_shift_pressure) are
computed from the Kalman-FILTERED STATE itself, which has no direct
ARIMA analog the way an innovation series does.

This script reuses the EXISTING 11-feature computation
(`lsc.diagnostics.features.compute_features`, the frozen COMPOSITE_V1
include-list) and the existing composite detector factory
(`lsc.eval.detectors.make_composite_detector`) completely unmodified --
the only change is the model fed to them: `lsc.models.ARIMAModel`
(AIC-selected ARIMA fit on the training prefix, forward-filtered;
`fittedvalues` as the state-analog, standardized residuals as the
innovations-analog) in place of `lsc.models.KalmanModel`. Same
calibration pipeline, same seeds, same per-time standardization, same
max-score composite rule as the real Kalman composite.

Judgment calls (disclosed, not assumed):
  - break_pressure, variance_pressure, variance_pressure_slow,
    variance_quiet, innovation_ac (5/11 features) act on `innovations`
    only -- DIRECT substitution, already precedented: arima_cusum_score
    and arima_var_cusum_score in this repo already run break_pressure /
    variance_pressure|quiet on these exact ARIMA standardized
    residuals, so this is not a new assumption for those five.
  - level_change, slope, acceleration, instability, persistence,
    state_shift_pressure (6/11 features) act on `filtered` -- JUDGMENT
    CALL: ARIMA's one-step-ahead fitted value (its conditional-mean
    forecast of Y itself) stands in for the Kalman filtered state (the
    model's belief about a literally separate latent AR(1) variable).
    Mechanically substitutable (both are length-T causal real-valued
    paths with the same warmup/NaN handling), but NOT the same object:
    an ARIMA model has no state distinct from the series it fits, so
    "filtered state slope" becomes "one-step-ahead forecast slope" --
    a real interpretive narrowing, reported here rather than assumed
    away. No feature was dropped or approximated beyond this
    substitution; all 11 got a direct feed of (filtered, innovations).

Grid: same core ladder cells as Table 3 -- channel in {r, q} x
vol_mult in {1.5, 3.0} x SNR in {0.1, 0.5, 2.0}, phi=0.95, T=500,
n_train=125, n_reps=500, FAR=0.05 -- with existing raw / ARIMA-CUSUM /
composite-on-Kalman numbers pulled from the already-published grid_v1
(r-channel composite), grid_v4_varbench_core (raw/ARIMA, r-channel) and
grid_v5_qbreak (raw/ARIMA/composite, q-channel) results, not
recomputed.

Usage: python experiments/exp20_composite_on_arima.py [n_reps]
Output: prints all four columns (raw, ARIMA-CUSUM, composite-on-Kalman,
composite-on-ARIMA) per cell; writes
paper_assets/exp20_composite_on_arima.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lsc.dgp import AR1StateDGP, BreakSpec
from lsc.diagnostics.alarms import calibrate, empirical_far
from lsc.diagnostics.features import COMPOSITE_V1
from lsc.eval.detectors import make_composite_detector
from lsc.models import ARIMAModel

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "paper_assets" / "exp20_composite_on_arima.csv"

PHI, R, T, N_TRAIN = 0.95, 1.0, 500, 125
FAR = 0.05
SEED_CAL, SEED_EVAL, SEED_FAR_CHECK, SEED_SCALE = 100_000, 200_000, 300_000, 900_000

CHANNELS = ("r", "q")
VOL_MULTS = (1.5, 3.0)
SNRS = (0.1, 0.5, 2.0)

# where the already-published raw / ARIMA / composite-on-Kalman numbers live
PUBLISHED = {
    "r": dict(
        raw_arima_path=REPO_ROOT / "paper_assets" / "grid_v4_varbench_core_results.csv",
        composite_path=REPO_ROOT / "paper_assets" / "grid_v1_results.csv",
        scenario_fmt="variance_x{vm}",
    ),
    "q": dict(
        raw_arima_path=REPO_ROOT / "paper_assets" / "grid_v5_qbreak_results.csv",
        composite_path=REPO_ROOT / "paper_assets" / "grid_v5_qbreak_results.csv",
        scenario_fmt="qvar_x{vm}",
    ),
}


def _published_rate(path: Path, arena: str, scenario: str, method: str) -> float:
    df = pd.read_csv(path)
    m = df[(df.arena == arena) & (df.scenario == scenario) & (df.method == method)]
    if not len(m):
        return float("nan")
    return float(m.detect_rate.iloc[0])


def run_cell(snr: float, channel: str, vol_mult: float, n_reps: int) -> dict:
    q = snr * (1 - PHI**2) * R
    null_dgp = AR1StateDGP(phi=PHI, q=q, r=R)

    fn = make_composite_detector(lambda: ARIMAModel(), null_dgp, T, N_TRAIN,
                                 n_scale_reps=min(50, n_reps),
                                 scale_seed0=SEED_SCALE, include=COMPOSITE_V1)
    det = calibrate("composite_on_arima", fn, null_dgp, T, n_reps=n_reps,
                    far=FAR, seed0=SEED_CAL)
    far_emp = empirical_far(det, null_dgp, T, n_reps=n_reps, seed0=SEED_FAR_CHECK)

    kind = "variance" if channel == "r" else "state_var"
    break_dgp = AR1StateDGP(phi=PHI, q=q, r=R,
                             breaks=[BreakSpec(kind=kind, time_frac=0.5, vol_mult=vol_mult)])
    break_time = break_dgp.breaks[0].time(T)
    paths = [break_dgp.sample(T, seed=SEED_EVAL + i).Y for i in range(n_reps)]
    detect_arima_composite = float(np.mean(
        [det.alarm_time(Y) is not None and det.alarm_time(Y) >= break_time for Y in paths]))

    arena = f"ar1_snr{snr}"
    conf = PUBLISHED[channel]
    # scenario keys in the published configs are "..._x3" (no decimal) and
    # "..._x1.5" (with decimal) -- NOT "..._x3.0".
    scen_vm = "3" if vol_mult == 3.0 else "1.5"
    scenario = conf["scenario_fmt"].format(vm=scen_vm)
    detect_raw = _published_rate(conf["raw_arima_path"], arena, scenario, "raw_var_cusum")
    detect_arima = _published_rate(conf["raw_arima_path"], arena, scenario, "arima_var_cusum")
    detect_composite_kalman = _published_rate(conf["composite_path"], arena, scenario, "lsc_composite")

    return dict(
        channel=channel, snr=snr, vol_mult=vol_mult, n_reps=n_reps,
        threshold_composite_arima=det.threshold,
        empirical_far_composite_arima=far_emp,
        detect_raw=detect_raw,
        detect_arima_cusum=detect_arima,
        detect_composite_kalman=detect_composite_kalman,
        detect_composite_arima=detect_arima_composite,
    )


def _load_existing() -> pd.DataFrame:
    if OUT_PATH.exists():
        return pd.read_csv(OUT_PATH)
    return pd.DataFrame(columns=["channel", "snr", "vol_mult", "n_reps"])


def _already_done(existing: pd.DataFrame, channel: str, snr: float,
                   vol_mult: float, n_reps: int) -> dict | None:
    if existing.empty:
        return None
    m = existing[(existing.channel == channel) & np.isclose(existing.snr, snr)
                 & np.isclose(existing.vol_mult, vol_mult) & (existing.n_reps == n_reps)]
    return m.iloc[0].to_dict() if len(m) else None


if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    t0 = time.time()
    existing = _load_existing()
    rows = []
    for channel in CHANNELS:
        for vol_mult in VOL_MULTS:
            for snr in SNRS:
                cached = _already_done(existing, channel, snr, vol_mult, n_reps)
                if cached is not None:
                    rows.append(cached)
                    print(f"[{time.time()-t0:6.0f}s] channel={channel} SNR={snr} "
                          f"vol_mult={vol_mult}: reused from existing "
                          f"{OUT_PATH.name}", flush=True)
                    continue
                out = run_cell(snr, channel, vol_mult, n_reps)
                rows.append(out)
                print(f"[{time.time()-t0:6.0f}s] channel={channel} SNR={snr} "
                      f"vol_mult={vol_mult} n_reps={n_reps}: "
                      f"thr={out['threshold_composite_arima']:.3f} "
                      f"FAR={out['empirical_far_composite_arima']:.3f} | "
                      f"raw={out['detect_raw']:.3f} "
                      f"arima_cusum={out['detect_arima_cusum']:.3f} "
                      f"composite_kalman={out['detect_composite_kalman']:.3f} "
                      f"composite_arima={out['detect_composite_arima']:.3f}", flush=True)
    df = pd.DataFrame(rows).sort_values(["channel", "vol_mult", "snr"]).reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[{time.time()-t0:6.0f}s] wrote {OUT_PATH}")
