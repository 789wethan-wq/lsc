# LSC repro pack (M7): `make all` regenerates every table and figure
# from scratch with pinned seeds. Requires the venv: `make venv` first
# (or point PY at any python with the package installed).
#
# Everything except `fred` is fully deterministic (seeds live in
# configs/ and experiment headers). `fred` depends on the live FRED
# download and re-runs on demand: `make fred`.

PY := .venv/bin/python
ASSETS := paper_assets

.PHONY: all venv test figures recovery exp01 exp02 exp03 exp04 exp05 exp06 exp08 grid grid_v2 grid_v4 fred paper clean

all: test figures recovery exp01 exp02 exp03 exp04 grid grid_v2 exp05 exp06 grid_v4 exp07 grid_v5 grid_v6 grid_v7 grid_v8 exp08 arl
	@echo "== repro pack complete (run 'make fred' separately: needs network) =="

venv:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

test:
	$(PY) -m pytest tests/ -q

figures:
	$(PY) -m lsc.plots.sample_paths $(ASSETS)/dgp_sample_paths.png

recovery:
	$(PY) experiments/m2_param_recovery.py 200

exp01:
	$(PY) experiments/exp01_idea_test.py 500
	$(PY) experiments/exp01_analyze.py

exp02:
	$(PY) experiments/exp02_snr_sweep.py 500

exp03:
	$(PY) experiments/exp03_persistence.py 500
	$(PY) experiments/exp03b_quietness.py 500

exp04:
	$(PY) experiments/exp04_multibreak.py 500

grid:
	$(PY) -m lsc.eval.runner configs/grid_v1.yaml

grid_v2:
	$(PY) -m lsc.eval.runner configs/grid_v2_T.yaml
	$(PY) -m lsc.eval.runner configs/grid_v2_misspec.yaml

exp05:
	$(PY) -m lsc.eval.runner configs/grid_v3_robust.yaml
	$(PY) -m lsc.eval.runner configs/grid_v3b_exceedance.yaml
	$(PY) -m lsc.eval.runner configs/grid_v3c_tail.yaml

exp06:
	$(PY) experiments/exp06_theory_check.py 1000

# M1 (R1): ARMA(1,1) equivalence gate — ARIMA vs Kalman innovation
# correlation on null paths (decision rule pre-registered in CHANGELOG)
exp07:
	$(PY) experiments/exp07_arma_equivalence.py 200

# M2 (R1): q-break (state-innovation variance) ladder — the second
# break channel. Runs after grid_v4 so qbreak_ladder.py can extend the
# r-channel ladder_table.csv with the q rows (break_channel column).
grid_v5:
	$(PY) -m lsc.eval.runner configs/grid_v5_qbreak.yaml
	$(PY) experiments/qbreak_ladder.py

# M3 (R1): phi sweep — mu_inf vs innovation-CUSUM detection (fast-or-never
# boundary). Headline theory-verification figure.
grid_v6:
	$(PY) -m lsc.eval.runner configs/grid_v6_phisweep.yaml
	$(PY) experiments/phisweep_analyze.py

# M4 (R1): local-level (RW-state) arena — level degeneracy + variance
# whitening-mandatory demonstration (replaces the old one-clause dismissal)
grid_v7:
	$(PY) -m lsc.eval.runner configs/grid_v7_llevel.yaml
	$(PY) experiments/llevel_analyze.py

# M7 (R1b): phi x q-break cross-grid — does raw's q-break advantage track
# the 1/(1-phi^2) amplification? (pre-registered before the run)
grid_v8:
	$(PY) -m lsc.eval.runner configs/grid_v8_phiqbreak.yaml
	$(PY) experiments/phiqbreak_analyze.py

# P2: PELT (offline changepoint) calibrated to a matched FAR, then
# evaluated on offline localization rather than delay
exp08:
	$(PY) experiments/exp08_pelt_benchmark.py 300

# M5 (R1): ARL0/ARL1 vocabulary from existing FAR tables + parquets
arl:
	$(PY) experiments/arl_report.py

# varbench addendum: whitening-ladder benchmarks (decision rule
# pre-registered in experiments/CHANGELOG.md before first run)
grid_v4:
	$(PY) -m lsc.eval.runner configs/grid_v4_varbench_core.yaml
	$(PY) -m lsc.eval.runner configs/grid_v4_varbench_T.yaml
	$(PY) experiments/varbench_ladder.py

fred:
	$(PY) experiments/m6_fred.py 200

# SSRN working-paper PDF (M5): pandoc + tectonic build of the patched
# draft -> paper_assets/lsc_wp.pdf. Requires tectonic on PATH.
paper:
	$(PY) experiments/build_paper.py

# m6x real-data extension: uses pinned snapshots in data/ (no network
# needed except realtime, which reads/caches ALFRED vintages)
realdata:
	$(PY) experiments/real_data.py indpro 200
	$(PY) experiments/real_data.py indpro 200 --train 180 --monitor 36 --tag _w180
	$(PY) experiments/real_data.py indpro 200 --far 0.01 --tag _far1
	$(PY) experiments/real_data.py indpro 200 --far 0.10 --tag _far10
	$(PY) experiments/real_data.py indpro 200 --far 0.20 --tag _far20
	$(PY) experiments/real_data.py gdp 200
	$(PY) experiments/real_data.py gs10 200
	$(PY) experiments/real_data.py unrate 200
	$(PY) experiments/real_data_eval.py

realtime:
	$(PY) experiments/realtime_check.py 200

clean:
	rm -f $(ASSETS)/*.png $(ASSETS)/*.csv $(ASSETS)/*.parquet \
	      $(ASSETS)/*.tex $(ASSETS)/*.log
