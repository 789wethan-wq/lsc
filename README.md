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
| M7 Repro pack | done — `make all` regenerates tests, tables, figures with pinned seeds (`make fred` separately: network) |

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
make all         # tests + every table/figure, pinned seeds (hours)
make fred        # M6 real-data application (needs network)
```

Individual pieces: `make test | figures | recovery | exp01 | exp02 | exp03 |
exp04 | grid | grid_v2`. Headline findings and every negative result are in
`experiments/FINDINGS.md`; all post-hoc design changes (with pre-registered
hypotheses for each experiment round) are in `experiments/CHANGELOG.md`.

Known v2+ directions (not implemented): switching-SSM (Kim filter) model
layer; adaptive composite weighting to reduce the breadth tax.
