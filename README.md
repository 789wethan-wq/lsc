# LSC — Latent-State Structural Change Detection

Research codebase for the two-layer framework in `../SPEC_latent_state_change.md`:
a **state-estimation layer** (Kalman / HMM / Markov-switching) that infers a
latent state from noisy observations, and a **diagnostics layer** that turns
features of the *filtered* (strictly causal) state path into calibrated alarms
for hidden structural change.

## Status

| Milestone | State |
|---|---|
| M1 DGP engine | done — `tests/test_dgp.py` green, figure sheet in `paper_assets/` |
| M2 Estimation layer | done — parameter recovery within 2 MC-SE (`paper_assets/m2_param_recovery.*`) |
| M3 Diagnostics layer | done — 11 features (incl. quietness/dynamics: `variance_quiet`, `variance_pressure_slow`, `innovation_ac`), per-time-point null standardization, `tests/test_no_lookahead.py` green (bit-identical) |
| M4 Benchmarks | done — raw-Y CUSUM, ARIMA+CUSUM, plain-HMM flip, offline PELT; parity test green |
| M5 Full simulation study | done — config-driven runner (`lsc/eval/runner.py`); v1 grid (`configs/grid_v1.yaml`) + v2 robustness: T sweep (`grid_v2_T.yaml`), misspecification arenas (`grid_v2_misspec.yaml`: t₅ noise, nonlinear drift), multi-break event-level F1 (`experiments/exp04_multibreak.py`), tail-robust exceedance detector (`grid_v3*.yaml`, exp05a–c); results in `experiments/FINDINGS.md` |
| Theory | done — fast-or-never formalized (innovation mean path, never-detect bound, Wald delay) in `experiments/THEORY.md` (`lsc/theory.py`), numerically verified by `experiments/exp06_theory_check.py` |
| M6 Real data | done — INDPRO (FRED) rolling causal alarms vs NBER reference dates, parametric-bootstrap calibration (`experiments/m6_fred.py`); m6x extension: pinned snapshots (`data/`), GDP + GS10 series, alarm attribution, window/FAR sensitivity, permutation tests (`real_data.py`, `real_data_eval.py`), ALFRED real-time vintage check of the GFC/COVID timing claims (`realtime_check.py`) |
| M7 Repro pack | done — `make all` regenerates tests, tables, figures with pinned seeds; `make realdata`/`make realtime` regenerate the real-data tables from pinned snapshots (no network needed); see "Regenerating Tables 1-7 and Appendix C" below |

## Key design decisions

- **No lookahead, enforced structurally.** Model parameters are estimated on a
  training prefix only (`Model.fit`), then a forward-only filter runs with
  fixed parameters (`Model.filter`). `tests/test_no_lookahead.py` perturbs
  `Y[t+1:]` and asserts bit-identical filtered states, innovations, all eleven
  diagnostic features, and all detector scores at times ≤ t, for every model.
  Detector scores are NaN during the training prefix — no online claims on
  data used for estimation.
- **Common calibration harness.** Every method (ours and benchmarks) gets its
  alarm threshold from the same routine (`lsc.diagnostics.alarms.calibrate`)
  on the same matched-null draws with the same budget; empirical FAR is
  re-verified on fresh nulls. Seed ranges for calibration / evaluation /
  FAR-check / feature-scales are disjoint (see `configs/exp01_idea_test.yaml`).
- **DGP and estimator modules share no code** (SPEC §4.3).
- Post-hoc design changes are logged in `experiments/CHANGELOG.md` (SPEC §11).

## Running

```bash
make venv        # python -m venv .venv && pip install -e ".[dev]"
make all         # tests + Tables 1-5 (+ most of Appendix C), pinned seeds (hours)
make realdata    # Tables 6-7 (+ Appendix C's rd_eval row) -- pinned snapshots in
                 # data/, no network needed
make realtime    # Appendix C's GFC/COVID real-time rows -- pinned vintages in
                 # data/vintages/, no network needed unless a vintage is missing
make paper       # rebuilds paper_assets/lsc_wp.pdf from PAPER_DRAFT.md
                 # (needs pandoc + tectonic on PATH)
```

`make fred` is the **older, superseded** `m6_fred.py` script (INDPRO only,
live-download-only) kept for provenance; it is not what the paper's Table 6/7
or Appendix C real-data rows come from — those all come from `make realdata`
(`real_data.py` / `real_data_eval.py`, the m6x extension), which is fully
reproducible from the pinned `data/` snapshots.

Individual pieces: `make test | figures | recovery | exp01 | exp02 | exp03 |
exp04 | grid | grid_v2`. Headline findings and every negative result are in
`experiments/FINDINGS.md`; all post-hoc design changes (with pre-registered
hypotheses for each experiment round) are in `experiments/CHANGELOG.md`.

Known v2+ directions (not implemented): switching-SSM (Kim filter) model
layer; adaptive composite weighting to reduce the breadth tax.

## Regenerating Tables 1-7 and Appendix C from scratch

Every command below is a `make` target already defined in `Makefile`; this
section only maps each published table to the exact command(s) and output
file(s) that produce it, in the order that matters (some later scripts read
and extend files an earlier script wrote).

```bash
make venv
make all         # runs grid, grid_v4, grid_v5, grid_v8, exp08, arl (and
                 # everything else `all` lists) in the order below
make realdata    # Tables 6-7
make realtime    # Appendix C real-time rows
make paper       # optional: rebuild the PDF itself
```

| Table | Command(s) (in order) | Output file(s) |
|---|---|---|
| Table 1 (ARL₀ by arena/method) | `make grid` (`configs/grid_v1.yaml`), then `make arl` (`experiments/arl_report.py`) | `paper_assets/arl_table.csv` |
| Table 2 (ARL₁: detect rate + mean delay) | same as Table 1 | `paper_assets/arl1_table.csv` |
| Table 3 (ladder, r + q channels) | `make grid_v4` (`grid_v4_varbench_core.yaml` + `_T.yaml`, then `experiments/varbench_ladder.py`), **then** `make grid_v5` (`grid_v5_qbreak.yaml`, then `experiments/qbreak_ladder.py`) — order matters: `qbreak_ladder.py` reads and extends the `ladder_table.csv` `varbench_ladder.py` just wrote | `paper_assets/ladder_table.csv` |
| Table 4 (φ×q amplification, Δ) | `make grid_v5` **before** `make grid_v8` (`grid_v8_phiqbreak.yaml`, then `experiments/phiqbreak_analyze.py`) — the SNR-swept grid_v5 numbers are overlaid on the φ-swept ones | `paper_assets/grid_v8_phiqbreak_summary.csv` |
| Table 5 (PELT localization rate) | `make exp08` (`experiments/exp08_pelt_benchmark.py`) | `paper_assets/exp08_pelt_results.csv` (+ `exp08_pelt_far_calibration.csv`) |
| Table 6 (real-data alarm summary) | `make realdata` (runs `real_data.py` for indpro/gdp/gs10/unrate, then `real_data_eval.py`) | `paper_assets/rd_eval.csv` |
| Table 7 (INDPRO FAR/window sensitivity) | same `make realdata` run (it also runs indpro at `--far 0.01/0.10/0.20` and `--train 180 --monitor 36 --tag _w180`) | `paper_assets/rd_eval.csv` (rows tagged by the FAR/window variant) |
| Appendix C | union of every command above, plus `make realtime` (`experiments/realtime_check.py`, GFC/COVID rows) | all of the above, plus `paper_assets/rd_realtime.csv` |

`experiments/real_data_date_boundaries.py` prints (and saves to
`paper_assets/real_data_date_boundaries.csv`) the exact train/test date
boundaries `real_data.py` uses for every segment of every series, read
directly off the same `SERIES` config and segmentation loop — useful for
checking Table 6/7's claims against the actual monitored windows rather than
a description of them:

```bash
.venv/bin/python experiments/real_data_date_boundaries.py            # all four series
.venv/bin/python experiments/real_data_date_boundaries.py indpro gdp # subset
```
