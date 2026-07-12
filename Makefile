# LSC repro pack (M7): `make all` regenerates every table and figure
# from scratch with pinned seeds. Requires the venv: `make venv` first
# (or point PY at any python with the package installed).
#
# Everything except `fred` is fully deterministic (seeds live in
# configs/ and experiment headers). `fred` depends on the live FRED
# download and re-runs on demand: `make fred`.

PY := .venv/bin/python
ASSETS := paper_assets

.PHONY: all venv test figures recovery exp01 exp02 exp03 exp04 exp05 exp06 grid grid_v2 grid_v4 fred paper clean

all: test figures recovery exp01 exp02 exp03 exp04 grid grid_v2 exp05 exp06 grid_v4
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
	$(PY) experiments/real_data.py indpro 200 --far 0.10 --tag _far10
	$(PY) experiments/real_data.py gdp 200
	$(PY) experiments/real_data.py gs10 200
	$(PY) experiments/real_data_eval.py

realtime:
	$(PY) experiments/realtime_check.py 200

clean:
	rm -f $(ASSETS)/*.png $(ASSETS)/*.csv $(ASSETS)/*.parquet \
	      $(ASSETS)/*.tex $(ASSETS)/*.log
