"""Config-driven Monte Carlo experiment runner (M5).

One YAML config fully determines an experiment: arenas (DGP + model
spec), scenarios (break specs), methods, T values, replication counts,
FAR target, and all seed bases (SPEC §4.4). Produces one tidy parquet
(+ CSV), a FAR-calibration table, LaTeX tables with MC standard errors
in parentheses, and frontier plots.

Usage: python -m lsc.eval.runner configs/grid_v1.yaml
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lsc.dgp import AR1StateDGP, BreakSpec, LocalLevelDGP
from lsc.diagnostics.alarms import calibrate, empirical_far
from lsc.eval.detectors import (
    make_arima_cusum_detector,
    make_composite_detector,
    make_innovation_cusum_detector,
    make_plain_hmm_detector,
    make_raw_cusum_detector,
    make_state_cusum_detector,
)
from lsc.eval.metrics import detection_outcome, summarize_detection
from lsc.models import KalmanModel

DGP_CLASSES = {"AR1StateDGP": AR1StateDGP, "LocalLevelDGP": LocalLevelDGP}


def build_dgp(arena_cfg: dict, breaks: list[BreakSpec]) -> object:
    cls = DGP_CLASSES[arena_cfg["dgp"]]
    params = {k: v for k, v in arena_cfg.items()
              if k not in ("dgp", "kalman_spec", "name")}
    return cls(**params, breaks=breaks)


def build_breaks(scen_cfg: dict) -> list[BreakSpec]:
    specs = scen_cfg if isinstance(scen_cfg, list) else [scen_cfg]
    return [BreakSpec(**s) for s in specs]


def build_detector(method: str, arena_cfg: dict, null_dgp, T: int,
                   n_train: int, n_scale_reps: int):
    spec = arena_cfg.get("kalman_spec", "llevel")
    if method == "lsc_composite":
        return make_composite_detector(lambda: KalmanModel(spec), null_dgp,
                                       T, n_train, n_scale_reps=n_scale_reps)
    if method == "lsc_composite_robust":
        from lsc.diagnostics.features import COMPOSITE_ROBUST
        return make_composite_detector(lambda: KalmanModel(spec), null_dgp,
                                       T, n_train, n_scale_reps=n_scale_reps,
                                       include=COMPOSITE_ROBUST)
    if method == "lsc_composite_robust2":
        from lsc.diagnostics.features import COMPOSITE_ROBUST2
        return make_composite_detector(lambda: KalmanModel(spec), null_dgp,
                                       T, n_train, n_scale_reps=n_scale_reps,
                                       include=COMPOSITE_ROBUST2)
    if method == "lsc_kalman_cusum":
        return make_innovation_cusum_detector(lambda: KalmanModel(spec), n_train)
    if method == "lsc_state_cusum":
        return make_state_cusum_detector(lambda: KalmanModel(spec), n_train)
    if method == "lsc_tail_cusum":
        from lsc.eval.detectors import make_tail_cusum_detector
        return make_tail_cusum_detector(lambda: KalmanModel(spec), n_train)
    if method == "raw_cusum":
        return make_raw_cusum_detector(n_train)
    if method == "arima_cusum":
        return make_arima_cusum_detector(n_train)
    if method == "plain_hmm":
        return make_plain_hmm_detector(n_train)
    raise ValueError(f"unknown method {method}")


def run(config_path: str) -> pd.DataFrame:
    cfg = yaml.safe_load(Path(config_path).read_text())
    name = cfg["experiment"]
    out_dir = Path(cfg.get("out_dir", "paper_assets"))
    n_reps = int(cfg["n_reps"])
    far_target = float(cfg["far_target"])
    seeds = cfg["seeds"]
    t0 = time.time()

    far_rows, rows = [], []
    for T in cfg["T_values"]:
        n_train = int(round(cfg["train_frac"] * T))
        for arena_name, arena_cfg in cfg["arenas"].items():
            null_dgp = build_dgp(arena_cfg, [])
            calibrated = {}
            for method in cfg["methods"]:
                fn = build_detector(method, arena_cfg, null_dgp, T, n_train,
                                    n_scale_reps=min(50, n_reps))
                det = calibrate(method, fn, null_dgp, T, n_reps=n_reps,
                                far=far_target, seed0=seeds["calibration"])
                far = empirical_far(det, null_dgp, T, n_reps=n_reps,
                                    seed0=seeds["far_check"])
                calibrated[method] = det
                far_rows.append(dict(T=T, arena=arena_name, method=method,
                                     threshold=det.threshold,
                                     far_target=far_target, far_empirical=far))
                print(f"[{time.time()-t0:7.0f}s] T={T} {arena_name} {method}: "
                      f"thr={det.threshold:.3f} FAR={far:.3%}", flush=True)

            for scen_name, scen_cfg in cfg["scenarios"].items():
                breaks = build_breaks(scen_cfg)
                dgp = build_dgp(arena_cfg, breaks)
                break_time = breaks[0].time(T)
                for method, det in calibrated.items():
                    outcomes = [
                        detection_outcome(
                            det.alarm_time(
                                dgp.sample(T, seed=seeds["evaluation"] + i).Y),
                            break_time, T)
                        for i in range(n_reps)
                    ]
                    summ = summarize_detection(outcomes)
                    rows.append(dict(T=T, arena=arena_name, scenario=scen_name,
                                     method=method, **summ))
                    print(f"[{time.time()-t0:7.0f}s] T={T} {arena_name} "
                          f"{scen_name} {method}: "
                          f"detect={summ['detect_rate']:.2f}", flush=True)

    far_df = pd.DataFrame(far_rows)
    df = pd.DataFrame(rows)
    far_df.to_csv(out_dir / f"{name}_far_calibration.csv", index=False)
    df.to_parquet(out_dir / f"{name}_results.parquet", index=False)
    df.to_csv(out_dir / f"{name}_results.csv", index=False)
    write_latex_tables(df, far_df, out_dir, name)
    write_frontier_plots(df, out_dir, name)
    print(f"[{time.time()-t0:7.0f}s] wrote {out_dir}/{name}_*", flush=True)
    return df


def _fmt(mean: float, se: float) -> str:
    return f"{mean:.2f} ({se:.2f})" if np.isfinite(se) else f"{mean:.2f}"


def write_latex_tables(df: pd.DataFrame, far_df: pd.DataFrame,
                       out_dir: Path, name: str) -> None:
    lines = []
    for (T, arena), g in df.groupby(["T", "arena"]):
        tab = g.assign(
            detect=g.apply(lambda r: _fmt(r.detect_rate, r.detect_rate_se), axis=1),
            delay=g.apply(lambda r: _fmt(r.mean_delay_censored,
                                         r.mean_delay_censored_se), axis=1),
        ).pivot_table(index="scenario", columns="method",
                      values=["detect", "delay"], aggfunc="first")
        lines.append(f"% T={T} arena={arena}\n" + tab.to_latex())
    (out_dir / f"{name}_tables.tex").write_text("\n".join(lines))
    (out_dir / f"{name}_far.tex").write_text(
        far_df.round(3).to_latex(index=False))


def write_frontier_plots(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for (T, arena), g in df.groupby(["T", "arena"]):
        scens = sorted(g.scenario.unique())
        fig, axes = plt.subplots(1, len(scens), figsize=(3.5 * len(scens), 3.5),
                                 sharey=True, squeeze=False)
        for ax, scen in zip(axes[0], scens):
            sub = g[g.scenario == scen]
            for _, r in sub.iterrows():
                ax.errorbar(r.mean_delay_censored, r.detect_rate,
                            xerr=r.mean_delay_censored_se,
                            yerr=r.detect_rate_se, fmt="o", label=r.method)
            ax.set_title(scen, fontsize=8)
            ax.set_xlabel("mean delay (cens.)", fontsize=8)
        axes[0][0].set_ylabel("detect rate")
        axes[0][-1].legend(fontsize=6)
        fig.suptitle(f"{name} — T={T}, {arena}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_frontier_T{T}_{arena}.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    run(sys.argv[1])
