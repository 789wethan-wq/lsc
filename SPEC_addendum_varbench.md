# SPEC — LSC Pre-Submission Addendum: Variance-Benchmark Closure + SSRN Package

**Repo:** existing LSC codebase (harness, 77 tests, CHANGELOG discipline all in place).
**Executor:** Claude Code. This is an addendum, not a rebuild — reuse the calibration harness, seed-block layout, and metrics stack unchanged. Work milestones in order; do not regenerate paper assets until M4's decision rule has been applied.

---

## 0. Purpose

The paper's central framing ("the latent layer reads second moments that raw-data detectors do not") is currently scoped to "level-oriented detectors standard in this literature" because the benchmark set contains no raw-data variance detector. This spec closes that gap **before** journal submission and SSRN posting, then packages the paper. The design turns a referee defense into a positive result: instead of one missing benchmark, we run a **whitening ladder** — the same variance statistic at three levels of prewhitening — which localizes exactly where the latent layer's second-moment advantage comes from.

## 1. Critical rule: pre-register the claim-adoption logic (M0, before any new cell runs)

Write the following decision rule into `experiments/CHANGELOG.md` **before** running M2, with a commit timestamp. The paper will adopt whichever claim the results select; no post-hoc reinterpretation.

Let D_raw = raw variance CUSUM detection rate at ×1.5, T = 500, per SNR; D_comp = composite's published 0.82/0.87/0.91.

- **Outcome A (strong claim):** D_raw within 5 pp of FAR at every SNR → abstract/intro upgrade to "raw-data detectors, including a variance CUSUM given identical calibration, sit at chance."
- **Outcome B (prewhitening claim):** D_raw within 10 pp of D_comp at every SNR → reframe: the advantage is *prewhitening under autocorrelation*, not latency per se; §5 and §11 rewritten accordingly (the fast-or-never side is untouched).
- **Outcome C (mixed):** anything else → report the full ladder, keep the scoped language, add a paragraph explaining the SNR-dependence.

All three outcomes are publishable. Log which fired.

## 2. New detectors (M1)

Implement in `lsc/benchmarks/variance.py`, behind the existing benchmark interface:

1. **`raw_var_cusum`** — Page CUSUM of z_t² − 1, where z_t = (Y_t − ȳ_train)/σ̂_train, mean and SD frozen from the training prefix. Up-arm allowances k = 0.25 and k = 0.05 (mirror the latent variance-pressure features exactly); down-arm (quieting) CUSUM of 1 − z_t², k = 0.05. No per-time-point standardization (standalone detector, same treatment as the exceedance detector).
2. **`arima_var_cusum`** — identical statistic on ARIMA residuals from the existing ARIMA benchmark's fitted model (training-prefix fit, frozen). This is the middle rung: whitened, but not state-aware.
3. (Existing, no work) latent e²-based CUSUMs — the top rung.

The ladder is: raw → ARIMA-whitened → Kalman-whitened. Same statistic, same allowances, same calibration routine, three information sets.

**Tests (extend suite):** `test_no_lookahead` for both new detectors (bit-identical future-perturbation check); `test_benchmark_parity_harness` inclusion (identical calibration budget and seed blocks); training-freeze test (perturbing monitoring-period data must not change ȳ_train, σ̂_train, or ARIMA parameters).

## 3. Simulation cells (M2)

Run both new detectors through the existing runner on:

- **Core:** noise-scale ×1.5, ×3, ×⅔ at SNR {0.1, 0.5, 2.0}, T = 500. 500 reps, standard seed blocks.
- **T sweep:** ×1.5 at T ∈ {200, 2000} (comparability with the published 0.11/0.87/0.99).
- **Misspecification:** ×1.5 under t₅ noise. Prediction to test: raw_var_cusum should collapse like the composite did (0.87 → 0.16), since z² has the same tail sensitivity — if it does, the exceedance detector's §6 story extends to the raw side and gets stronger.
- **Level shifts (cheap, optional but run it):** 1σ and 3σ at SNR 0.5 — establishes the variance detectors don't accidentally detect levels, completing the disjoint-channels table.
- **Persistence cells** at SNR {0.5, 2.0} — the ladder's behavior near the information floor.

Output: one new parquet, `grid_v4_varbench`, same tidy schema. Report empirical FARs alongside (T = 200 hot-calibration check applies to the new detectors too).

## 4. Real-data rerun (M3)

Add `raw_var_cusum` (both arms) to the INDPRO, GDPC1, and GS10 monitoring runs — identical windows (120/60), identical per-segment bootstrap calibration at 5% FAR, pinned snapshots unchanged. Report: alarm dates, feature attribution, NBER-association permutation p (INDPRO), and whether it catches 2008/2020 on INDPRO and GDP and Volcker on GS10. **This can weaken the uniqueness claim on real data — that is the point of running it.** If raw_var_cusum catches the same crises, §10's claim becomes about the *simulation-calibrated subtlety threshold* (×1.5 invisibility), not real-data uniqueness; patch accordingly under the M0 rule's spirit and log it.

Do NOT rerun ALFRED vintages unless raw_var_cusum alarms on revised INDPRO — if it does, run the vintage check for it too (the GFC/COVID real-time comparison must be apples-to-apples).

## 5. Claim resolution and prose patch (M4)

Apply the M0 decision rule. Then patch `LSC_paper_v1.md` (working copy in repo; do not touch the original):

- Abstract: one sentence swaps per outcome A/B/C.
- §1 contribution 2 and the bracketed scoping note: resolve and delete the bracket.
- §5: add the ladder table (raw / ARIMA / latent × {×1.5, ×3, ×⅔} × SNR) and one paragraph interpreting it; delete the editorial note.
- §9.2: add the t₅ raw_var result (one sentence).
- §10: add raw_var rows to the real-data comparison; adjust the uniqueness language per M3.
- §11: update the practical recipe if outcome B (recipe becomes "whiten, then run the variance CUSUM"; the latent state estimate itself may be optional for second moments — say so plainly if true).
- CHANGELOG entry: which outcome fired, with the numbers.

## 6. SSRN package (M5)

- `make paper`: pandoc (or LaTeX via pandoc) build of the patched draft → `paper_assets/lsc_wp.pdf`. Title page: title, author (Ethan Wuang, E.W. Research), date, abstract, JEL codes [C12, C22, C52 — verify], keywords. Number the propositions; move derivation pointers into a proper appendix reference.
- Repro statement in the PDF references the public repo (confirm the repo is/will be public before the PDF claims it; otherwise phrase as "available on request").
- Final gate: `make all` green from a clean clone, all tests (77 + new) passing, `grid_v4_varbench` reproducible from pinned seeds.
- Deliverables to outputs: `lsc_wp.pdf`, the patched draft, the ladder table as standalone CSV.

## 7. Milestones & acceptance

- **M0** — decision rule committed to CHANGELOG *before any M2 run*. Accept: commit exists, timestamped, no grid_v4 artifacts predate it.
- **M1** — detectors + tests. Accept: full suite green including new no-lookahead and freeze tests.
- **M2** — grid_v4_varbench complete with MC-SEs and empirical FARs.
- **M3** — real-data reruns complete; comparison table generated.
- **M4** — outcome logged; prose patched; no bracketed editorial notes remain in the working draft.
- **M5** — PDF builds from clean clone; visual check of title page, propositions, tables.

## 8. Honest-outcome clause (inherited)

If outcome B fires — the latent state layer is unnecessary for second moments and the advantage is just prewhitening — the paper reports that as a finding and the title still holds: the answer to "when does filtering help you see a break?" becomes "for whitening, not for state estimation," which is arguably the sharper result. Do not soften it.
