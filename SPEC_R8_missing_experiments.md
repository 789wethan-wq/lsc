# Missing-Experiments Specification
### Response package for the review of *When Does Filtering Help You See a Break?*

**Scope.** Eight primary experiments (`exp44`–`exp51`) plus four tier-2 items (`exp52`–`exp55`),
each specified to the level where it can be implemented without further design decisions.
Every entry states the review point it answers, the exact construction, the seed allocation,
the output schema, a pre-registered prediction, and a decision rule that fixes in advance what
each possible outcome does to the manuscript.

**Read §0 first.** Four standing requirements apply to every experiment below and two of them
fix defects that have already cost this project real work.

---

## §0. Standing requirements

### 0.1 Persist per-replicate outcomes

`exp19` had to *reconstruct* per-replicate outcomes for Table 4 via a determinism argument
because the original run persisted only aggregates. That reconstruction is the weakest link in
the paired-SE story ("the pairing itself is inferred, not independently confirmed"). It must
never recur.

Every experiment below writes, alongside its aggregate CSV, a per-replicate long-format file:

```
paper_assets/exp<NN>_perrep.csv
  rep_id, arena, scenario, channel, vol_mult, snr, phi, detector,
  detected (0/1), alarm_index (int or NA), score_max, threshold, seed
```

This makes every paired SE, every correlation between detectors, and every re-analysis a
groupby rather than a re-run. Add a test that asserts the per-replicate file's aggregate
matches the aggregate file exactly.

### 0.2 Seed allocation

Existing blocks are in use and must not be touched: `100000+` (calibration), `110000+` (AR(2)
calibration), `200000+` (evaluation), `210000+` (AR(2) evaluation), `300000+` (fresh-null FAR
check), `400000+` (oracle / break-aware).

New allocation, 1000 apart per experiment:

| Experiment | Calibration | Evaluation | Fresh-null FAR |
|---|---|---|---|
| exp44 | **reuse grid_v4/grid_v5** | **reuse grid_v4/grid_v5** | 300000+ |
| exp45 | 100000–104999 (extended) | 200000–201999 (extended) | 300000–301999 |
| exp46 | 100000–104999 | 200000–200499 | 330000–331999 |
| exp47 (sim gate) | n/a | 232000+ | n/a |
| exp48 | 133000+ | 233000+ | 333000+ |
| exp49 | 134000+ | 234000+ | 334000+ |
| exp50 | 135000+ | 235000+ | 335000+ |
| exp51 | reuse Tables 2/3 | reuse Tables 2/3 | n/a |

**exp44 is the exception that matters.** It must run on the *identical* evaluation paths as the
published `raw_var_cusum` and `arima_var_cusum` rungs, or the whole point — a paired comparison
against them — is lost. Do not allocate it new evaluation seeds.

Within any experiment, all detectors share evaluation seeds. Across experiments, blocks are
disjoint. Assert both in tests.

### 0.3 Pre-register before running

Log each experiment's prediction and decision rule in `experiments/CHANGELOG.md` **before**
the first run, using the template in §12. This is the project's own established practice and
it is what lets a falsified prediction be reported as a finding rather than looking like a
post-hoc excuse. Two of the experiments below (exp47, exp48) are genuinely likely to come back
negative; pre-registration is what makes that publishable.

### 0.4 Multiplicity accounting

Appendix A's argument — that simulation-side follow-ups are heterogeneous objects and not a
comparable test family — covers exp44, exp45, exp46, exp48, exp49, exp50, exp51, and all of
tier 2. It does **not** cover exp47, which is a formal hypothesis test on the same real series
as §9. exp47's tests join §9's corrected family and expand it from 34 to 34 + *k*, where *k*
is the number of (series × event) channel tests actually run. Fix *k* in advance and state the
adjusted Bonferroni/BH thresholds in the pre-registration entry, not after seeing results.

---

## §1. Priority and dependency order

```
P0  exp52  Prop-1 bound evaluation          minutes, no simulation
    exp44  estimated-parameter Kalman rung   hours
    exp46  symmetric FAR recalibration       hours
              │
P1            ├── exp45  uniform n_reps increase        days (run after 44/46 so
              │                                          the new rung and corrected
              │                                          thresholds get the larger n)
    exp47  real-data channel identification  days (gated — see §5)
              │
P2  exp48  per-arm standardized windowed combination
              │
    exp49  q-dominant mixture  ◄──────────── depends on exp48 (adds the fixed
              │                              combination as a fifth rule)
    exp51  delay distributions               reuses persisted data from 45
              │
P3  exp50  PELT with variance cost + Bai–Perron
    exp53–55  tier 2
```

**If only three can be run: exp52, exp44, exp47.** Those three decide whether Proposition 1
must be restated, whether the §5 2×2 is a measurement or an assumption, and whether the
paper's application-relevance claim survives. The rest sharpen; these three determine what
the paper is allowed to say.

---

## §2. exp44 — Estimated-parameter Kalman variance CUSUM

**Answers:** Major Weakness 2; Question 4. The §5 2×2 currently fills the
`Kalman filter × single 3-arm CUSUM` cell with the ARIMA rung's numbers "by the ARMA(1,1)
equivalence." That equivalence is exact only at steady state with known parameters. Under the
paper's actual operating condition the rungs correlate at ρ̄ = 0.99 and AIC selects the true
(1,0,1) order only 7.8–12.0% of the time at φ = 0.95. A max-over-arms CUSUM calibrated at the
95th percentile of a null maximum is a function of tail excursions, not of median correlation.

**Construct.** `lsc.benchmarks.variance.est_kalman_var_cusum_score`.

Implement by *parameterizing* the existing `known_kalman_var_cusum_score` rather than
duplicating it — the only difference is the parameter source:

- fit (φ, q, r) by MLE on the training prefix (first 125 of 500 obs), identical to every other
  estimated rung;
- forward-filter with frozen parameters, diffuse initialization, scores NaN on training data;
- standardized one-step innovations `e_t`;
- the **identical** three-arm statistic used by `raw_var_cusum` and `arima_var_cusum`: up-arm
  Page CUSUMs of `e²−1` at k = 0.25 and k = 0.05, down-arm CUSUM of `1−e²` at k = 0.05, max
  over arms, **no per-time standardization**;
- same calibration routine, same FAR target, same matched-null seed block.

**Grid.** The 12 cells of Tables 3/5: channel ∈ {r, q} × vol_mult ∈ {1.5, 3} × SNR ∈ {0.1, 0.5,
2.0}, T = 500, n_train = 125, φ = 0.95, n_reps = 500 (raise to 2000 under exp45).

**Outputs.** `paper_assets/exp44_est_kalman_rung.csv`:

| column | note |
|---|---|
| `channel`, `vol_mult`, `snr` | cell |
| `detect_est_kalman`, `detect_arima`, `detect_raw` | last two copied from published grids, must match exactly |
| `delta_kalman_arima` | paired difference |
| `se_paired` | from the per-replicate join — **not** an independence bound |
| `outcome_corr` | per-replicate correlation between the two rungs' detect indicators |
| `threshold_est_kalman`, `threshold_arima` | |
| `far_fresh` | on the 300000+ block, 500 draws |

Plus a second file `exp44_innovation_tails.csv` reporting, per cell, the distribution of the
*statistic* gap rather than the series correlation: 50th/95th/99th percentile of
`|e_kalman,t − e_arima,t|` over all t and reps, and the correlation between the two rungs'
per-replicate **max scores** (the quantity the threshold actually acts on).

**Pre-registered prediction.** H44: `|Δ| ≤ 0.03` in ≥ 10 of 12 cells.

**Decision rule.**
- **A (H44 holds):** replace the 2×2's asserted cell with the measured one and change
  "identical by the ARMA(1,1) equivalence" to "identical to within MC error under estimation,
  measured." The claim is then earned rather than assumed and the paper is strictly stronger.
- **B (any cell `|Δ| > 0.10`):** the equivalence does not transfer to the estimated statistic.
  Report `est_kalman` as a distinct fourth rung throughout §5, and rewrite the §5 headline —
  "the ladder is really raw vs. whitened" becomes true only at the population level, with the
  estimated ladder having three distinct rungs. Also revisit the φ = 0.99 discussion, which
  currently attributes the r-channel reversal entirely to ARIMA's estimation fragility; if
  estimated Kalman also degrades there, that attribution is wrong.
- **C (in between):** report both rungs, describe as "close but not identical under
  estimation," and quantify with the tail file.

**Cost.** Low. No ARIMA refitting (the 8.8-hour exp20 pathology came from ARIMA fits, not
Kalman ones). The MLE fit is the same one the composite already performs; cache and reuse per
(seed, arena) if the harness allows.

**Gotcha.** The three-arm statistic must be byte-for-byte the same code path as
`raw_var_cusum`/`arima_var_cusum` — pass a different input series to one shared function.
A reimplementation, even a faithful one, would make any observed difference uninterpretable.

---

## §3. exp45 — Uniform n_reps increase on the core grids

**Answers:** Minor Weakness 1, and roughly a third of the manuscript's length. A large share
of the current text adjudicates 1–3pp differences against 1.5–2.5pp paired SEs (Table 2b's six
negative cells, the φ = 0.95/0.97 plateau, exp14's SNR-0.5 gap, exp40's 46-cell pairing
audit). At n = 2000 most of those questions answer themselves and the surrounding prose can be
deleted.

**The current justification for not doing this does not hold.** §10 declines on the grounds
that re-running "the load-bearing cells" would introduce a multiplicity problem — which is
correct, and is an argument against *selective* re-runs only. A uniform increase across all
cells of a grid has no multiplicity implication whatever. Pre-register the uniformity
explicitly so the distinction is on record.

**Construct.**

| Grid | current n_reps | target | current calib reps | target |
|---|---|---|---|---|
| `grid_v1` (level ladder, Tables 1–2) | 500 | 2000 | 500 | 5000 |
| `grid_v4_varbench` (r channel) | 500 | 2000 | 500 | 5000 |
| `grid_v5_qbreak` (q channel) | 500 | 2000 | 500 | 5000 |
| `grid_v8_phiqbreak` (Table 4) | 500 | 2000 | 500 | 5000 |
| `grid_v9b` / `r_phi_sweep` (Table 3c) | 500 | 2000 | 500 | 5000 |
| exp26 known-parameter variance (Table 2b) | 500 | 2000 | 500 | 5000 |

Extend, do not replace, the seed blocks: reps 0–499 keep their existing seeds so the first 500
draws reproduce the published numbers exactly. This gives a free internal consistency check —
if the first-500 subset does not reproduce the published aggregate, something else changed.
Assert this in a test.

`exp38` already showed 5000 calibration reps moves out-of-sample FAR from 5.3/6.4/7.2% to
4.5/4.7/4.7% for `raw_cusum`; apply the same budget to every detector (this is also exp46's
requirement, which is why exp46 should be folded into the same run).

**Expected precision.** At n = 2000, worst-case unpaired SE is 0.011; paired SEs on the cells
currently at 0.014–0.025 should land near 0.007–0.013. The φ = 0.95-vs-0.99 subtle-Δ gap
(0.038, currently ≈1.5 SE) becomes ≈2.9 SE — resolvable either way.

**Decision rule.** Fixed before running, for every published ordering:
- reverses sign and clears 2 SE at n = 2000 → report as reversed, with the n = 500 value in a
  footnote;
- same sign, clears 2 SE → strengthen the prose from "suggestive" to stated;
- within 2 SE at n = 2000 → report as **unresolved at n = 2000** and delete the surrounding
  adjudication paragraphs rather than re-hedging them.

No cell is exempt; no cell is re-run again afterward.

**Cost.** ~4× current wall clock on those grids, plus 10× on calibration (which is cheaper per
rep). The ARIMA rungs dominate; consider capping `statsmodels` optimizer iterations with a
documented fallback and logging how often the cap binds, since exp20 documented a handful of
pathologically slow fits.

---

## §4. exp46 — Symmetric false-alarm-rate recalibration

**Answers:** Major Weakness 3; Questions 5 and 7. Table 1 shows empirical FAR from 3.4% to 8.2%
against a 5% target across detectors that are then compared on power. `exp38` corrected only
`raw_cusum` — the detector that calibrates *hot*, i.e. the one whose correction can only hurt
it. `lsc_kalman_cusum` calibrates *cold* (3.4% at SNR 0.1, 4.8% at SNR 2.0) and was never
loosened, so it is being compared at a stricter operating point than its rival. A calibrated-
parity harness is Contribution 1; it currently does not deliver parity within ±3pp.

**Construct, part A — reconcile the discrepancy first.** §2/Table 1 report `raw_cusum` FAR as
4.0 / 6.2 / 8.2%; `exp38` reports the same detector's baseline out-of-sample FAR as
5.3 / 6.4 / 7.2%. At SNR 0.1 these disagree on the *sign* of the miscalibration. Recompute
both under one fresh-null block and determine which is in-calibration-sample and which is
out-of-sample. Document the answer in the CHANGELOG and correct whichever table is stale.
This must be settled before any other number in exp46 is trusted.

**Construct, part B — recalibrate everything.** For every detector in Tables 1–3
(`raw_cusum`, `raw_var_cusum`, `arima_var_cusum`, `est_kalman_var_cusum` from exp44,
`lsc_kalman_cusum`, `lsc_state_cusum`, `lsc_composite`, `lsc_tail_cusum`), at every arena:

1. calibrate at 5000 reps (block `100000–104999`);
2. verify on 2000 **fresh** nulls (block `330000–331999`, disjoint from the standing 300000
   block so the existing GARCH check is unaffected);
3. if fresh FAR is outside [4.5%, 5.5%], iterate the threshold by bisection on the fresh block
   until it is, and report both the quantile threshold and the FAR-matched threshold;
4. re-run Tables 2 and 3 at the FAR-matched thresholds.

**Outputs.** `paper_assets/exp46_far_parity.csv`:

```
detector, arena, snr,
  thresh_q500, thresh_q5000, thresh_far_matched,
  far_fresh_q500, far_fresh_q5000, far_fresh_matched,
  detect_q500, detect_q5000, detect_matched,
  delta_detect_matched_minus_published
```

Report this as a replacement for Table 1, with a `far_fresh_matched` column that is inside
[4.5%, 5.5%] in every row. That table then genuinely earns the phrase "calibrated parity."

**Pre-registered prediction.** H46: with every detector FAR-matched, the level-3σ ordering
(raw > innovation) survives at every SNR, and the gap narrows by ≤ 0.10 at each.

**Decision rule.**
- H46 holds → Contribution 1 stands, and the paper can state the FAR-matched numbers as the
  headline instead of caveating Table 1.
- Gap narrows by > 0.10 at any SNR → "raw CUSUM dominates at every SNR" needs an explicit
  scope qualifier naming the calibration convention, in the same register §10 already uses for
  the one-sided/known-parameter tie.
- Ordering reverses anywhere → report as a reversal; leg (i) of the trichotomy becomes
  convention-dependent rather than robust, which is a substantially different paper and must
  be said in the abstract.

**Cost.** Moderate; fold into the exp45 run to avoid paying the simulation cost twice.

---

## §5. exp47 — Real-data channel identification (gated)

**Answers:** Major Weakness 4; Question 6. The claim that the Great Moderation and the
2008/2020 volatility events are **q-channel** (state-shock) rather than **r-channel**
(observation-noise) breaks is what makes leg (iii) the applied leg and leg (ii) the merely
technical one. The support offered is a citation to McConnell & Pérez-Quirós (2000) — which
documents a 1984Q1 break in GDP-growth volatility and traces it to durable-goods production
and the inventory share of durables, and contains no latent-state / measurement-noise
decomposition at all. §9's "attribution" is feature-level (`variance_pressure`,
`variance_quiet`), and an r-break and a q-break both fire `variance_pressure`. The bridge is
currently unsupported in both directions.

This experiment is **gated**: a simulation validation must pass before the real-data step runs
at all.

### 5.1 Gate — confusion matrix on known ground truth

Before touching FRED data, verify that the identification procedure can recover a channel it
*knows* exists.

- Simulate AR(1)+noise paths of length 2W with a break at the midpoint, at each of the paper's
  SNRs and both break sizes, in each channel — 12 cells, n = 500, seed block `232000+`.
  Use W = 120 (matching the monthly real-data window) and W = 40 (quarterly, for GDP).
- On each path, fit (φ, q, r) by MLE separately on `[0, W)` and `[W, 2W)`.
- Classify: `q-break` if `|log(q̂_post/q̂_pre)| > |log(r̂_post/r̂_pre)|`, else `r-break`.
- Report the 2×2 confusion matrix per (SNR, vol_mult, W).

**Gate condition:** correct-channel classification rate ≥ 0.70 at the relevant effect size and
window length. If the gate fails at ×1.5 but passes at ×3, the real-data step may proceed for
coarse events only (2008, 2020) and must not be run for the Great Moderation, whose effect
size is closer to subtle.

**If the gate fails everywhere, that is the deliverable.** "The channel is not identifiable
from a 120-month window at this SNR" is a real, reportable finding, and it is sufficient on
its own to require Major Weakness 4's resolution path (b): soften every application-relevance
claim to a conditional and say plainly that the antecedent is not established. Do not run the
real-data step to see what it says anyway.

### 5.2 Real-data step (only if the gate passes)

- **Series × events:** INDPRO × {1984-01, 2008-09, 2020-03}; GDPC1 × {1984Q1, 2008Q4, 2020Q2};
  UNRATE × {2008-09, 2020-03}; GS10 × {1979-10, 2008-12, 2022-03}. Fix *k* = the number of
  (series, event) pairs actually run, in advance, for §0.4's multiplicity ledger.
- **Windows:** symmetric `[τ−W, τ)` and `[τ, τ+W)`, W = 120 months / 40 quarters. Where the
  window would overlap another registered event, truncate symmetrically and record the
  truncation.
- **Statistics:** `Λ_q = log(q̂_post/q̂_pre)`, `Λ_r = log(r̂_post/r̂_pre)`.
- **Inference:** parametric bootstrap. Fit the model on the pooled `[τ−W, τ+W)` window, simulate
  B = 2000 no-break paths, refit pre/post on each, and take the joint null distribution of
  `(Λ_q, Λ_r)`. Report a marginal bootstrap p for each and the joint position of the observed
  pair in the null cloud.
- **Identification diagnostic — mandatory, not optional.** q and r are weakly separately
  identified in a short AR(1)+noise window near the unit root; the well-identified quantity is
  the ratio q/r. For every window, compute and store the **profile log-likelihood surface in
  (q, r)** on a grid, extract the 95% likelihood-ratio confidence region, and report whether
  the pre- and post-window regions are disjoint. Report `φ̂` and whether it hit the [0.01, 0.99]
  clip. A cell whose confidence regions overlap substantially is reported as *not identified*,
  regardless of what the point estimates say — this is the same discipline exp09/exp17 already
  applied to UNRATE, generalized.

**Framing.** This is a retrospective, break-time-known analysis, in the same oracle category as
the known-parameter columns of exp10/exp26/exp30 and the break-aware GARCH of exp37. Label it
that way explicitly. It is not a monitoring result and no real-time or vintage claim attaches
to it.

**Outputs.** `paper_assets/exp47_channel_id.csv`:

```
series, event, window_months, n_pre, n_post,
  phi_pre, phi_post, phi_clipped_pre, phi_clipped_post,
  q_pre, q_post, r_pre, r_post,
  lambda_q, lambda_r, p_boot_q, p_boot_r,
  ci_regions_disjoint (bool), identified (bool),
  channel_call (q | r | both | not_identified)
```
plus `exp47_gate_confusion.csv` and the profile-likelihood surfaces under `paper_assets/exp47_profiles/`.

**Pre-registered decision rule.**
- **A** — `Λ_q` significant, `Λ_r` not, in a majority of identified cells → q-channel confirmed.
  §5's application-relevance claim is earned; state it as a finding with its p-values rather
  than as an assertion, and add exp47's tests to §9's corrected family.
- **B** — `Λ_r` moves and `Λ_q` does not → the paper's leg ordering is inverted. Leg (ii)
  ("prewhitening wins") becomes the applied result and leg (iii) the technical one. This
  rewrites the abstract and §10's practical recipe, and would make always-ARIMA (exp18's
  pooled winner) the correct recommendation for the right reason rather than by accident.
- **C** — both move, or the identification diagnostic fails in most cells → withdraw the
  channel-relevance claim. Restate leg (iii) conditionally throughout and delete "the channel
  that matters for the volatility-regime events of the application" from Contribution 2.
- **D** — gate fails → as §5.1.

**Cost.** The gate is a day of compute; the real-data step is minutes plus the bootstrap. The
profile-likelihood surfaces are the expensive part and are worth it — they are what
distinguishes "we identified the channel" from "we produced two point estimates."

---

## §6. exp48 — Per-arm standardized windowed combination

**Answers:** Specific Comment 11; converts §7's negative result into a positive one. `exp36`
found that `windowed_combined` (max over the raw scores of `windowed_raw_cusum` and
`windowed_raw_var`, jointly calibrated) collapses `windowed_raw_var`'s 0.932 first-break recall
to 0.044 in the var→level ordering, because the mean-shift arm's heavier null tail sets the
shared threshold. §7 diagnoses this correctly and then states the fix — per-arm standardization,
"not attempted here." The fix is small and the composite already implements the machinery.

**Construct.** Three arms, evaluated on the identical scenarios, arena, seeds and re-arm
protocol as exp36 (`level_var` and `var_level`, breaks 150 obs apart, re-arm at half threshold
+ 20-obs refractory, n = 500, 5% FAR):

- **Arm A — `windowed_combined_std` (the proposed fix).** Standardize each windowed statistic
  at each time point *t* against its own null distribution at *t* (median and IQR over
  `N_null` null replicates), reusing `lsc.diagnostics.features`'s existing per-t standardization
  helper — do not reimplement it — then take the max of the two standardized series, then
  calibrate that max against the null-max distribution.
- **Arm B — split-budget baseline.** Two separate thresholds, each calibrated at 2.5% FAR
  (Bonferroni split of the 5% budget), alarm if either fires. This is the simpler competitor
  and it distinguishes "the max needed standardizing" from "two statistics just need a split
  budget." If B matches A, the paper should recommend B — it is trivially explainable.
- **Arm C — published `windowed_combined`**, re-run for reference.

Set and report `N_null`; the per-t scale estimates are themselves noisy and the null-max
threshold is sensitive to their tails.

**Outputs.** `paper_assets/exp48_windowed_std.csv` with, per (ordering, arm):
`recall_break1, recall_break2, precision, F1, far_fresh, threshold`, plus an
argmax-attribution column (which arm fired at alarm time) and the null-tail summary for each
standardized arm (95th/99th percentile of the standardized score) confirming the arms are now
on comparable scales.

**Pre-registered prediction.** H48: per-arm standardization recovers ≥ 0.80 of each arm's own
matched-channel single-arm recall in both orderings — i.e. var→level `recall_break1` ≥ 0.75
(against `windowed_raw_var` alone at 0.932) and level→var `recall_break1` ≥ 0.55 (against
`windowed_raw_cusum` alone at 0.692).

**Decision rule.**
- H48 holds → §7's open problem closes. Rewrite the section from "channel-matched multi-break
  monitoring under channel uncertainty is still open" to a solved case with the mechanism
  (shared max-thresholds require per-arm standardization) generalized as a lesson for the
  11-feature composite too.
- Recovery between 0.50 and 0.80 → partial fix; report both arms and the residual gap.
- Recovery < 0.50 → the dilution is **not** a scale artifact and §7's stated diagnosis is
  wrong. Report the falsification and revise the diagnosis; this is a more interesting outcome
  than the fix working, and it also puts §8.3(ii)'s "composite dilution" explanation in
  question, since it is the same claimed mechanism.

---

## §7. exp49 — Fixed-rule baselines under a q-dominant mixture

**Answers:** Specific Comment 17. §10's practical recommendation ("always-ARIMA is the better
fixed rule under channel uncertainty") is derived from exp14/exp18's **50/50** r/q mixture at
×1.5. §5 argues at length that the empirically relevant channel is q. Under a q-dominant
mixture the recommendation plausibly flips, and the paper would then be recommending against
its own applied conclusion.

**Construct.** Re-run the exp14/exp18 evaluation across a mixture grid:

- channel weight `w_q ∈ {0.5 (published), 0.8, 1.0}`;
- SNR weighting: equal thirds (published) and, as a second row, whatever mixture exp47's
  fitted real-data SNRs actually support — if exp47 does not run or does not identify, keep
  equal thirds and say so;
- break size ×1.5 (as published), and ×3 as a second panel, since the q-channel raw advantage
  is much larger at ×3 (Table 3: 0.72/0.96/0.96 vs 0.26/0.79/1.00) and the published mixture
  test only ever used the subtle break.

Fixed rules evaluated: `always-raw`, `always-ARIMA`, `jointly-calibrated combined` (one shared
5% budget — do not let each arm run at its own 5%), `oracle-best-per-SNR`, and — if exp48
succeeds — **`per-arm-standardized combined`** as a fifth rule. That link is the point: if
per-arm standardization fixes the §7 dilution, it should also remove the §10 "run both" tax,
and the two sections then tell one story instead of two disconnected negative ones.

**Outputs.** `paper_assets/exp49_mixture_rules.csv`:
`w_q, vol_mult, snr_weighting, rule, rate, se, delta_vs_best_fixed`.

**Pre-registered prediction.** H49: `always-raw` overtakes `always-ARIMA` at `w_q ≥ 0.8` on the
×1.5 panel, and at `w_q ≥ 0.5` on the ×3 panel.

**Decision rule.**
- H49 holds → §10's recipe becomes conditional on the analyst's prior over channels, with the
  crossover weight reported. That is a better recommendation than either current absolute.
- H49 fails → §10 stands as written, **and** §5's "the q channel is the one that matters"
  loses its practical consequence: even where raw wins the q channel cell-by-cell, ARIMA wins
  the decision problem. Report that narrowing explicitly; it is an honest and interesting
  result, not a defeat.

---

## §8. exp50 — PELT with a variance-sensitive cost, plus Bai–Perron

**Answers:** Specific Comment 12. §8.5 applies PELT with an `l2` (mean-shift) cost and then
concludes that off-the-shelf offline methods fail on variance breaks. `ruptures` ships
variance-sensitive costs; using the mean-shift cost and generalizing from it is a straw
comparison. §8.5 half-acknowledges this ("PELT's default `l2` cost model is a mean-shift
statistic") and then keeps the general conclusion.

**Construct.**
- Re-run exp08's localization benchmark with `model ∈ {"l2" (published), "normal", "rbf"}`,
  penalty calibrated by the same bisection-to-5%-FAR protocol on null AR(1) paths, same ±25-obs
  localization window, n = 500 (raise from 300 to match ICSS's grid so the three offline
  methods sit on one table).
- Extend the scenario set to include the q-channel variance breaks that ICSS covers, so PELT,
  ICSS, and the causal `raw_var_cusum` appear in a single comparison rather than two.
- Add **Bai–Perron** on the level scenarios (`ruptures` `Dynp` with known K = 1, or an
  equivalent dynamic-programming segmentation). §7 and Related Work both name Bai & Perron
  (1998, 2003) as the natural retrospective reference and neither benchmarks it; a single
  known-K run on the level ladder closes that gap cheaply.

**Outputs.** `paper_assets/exp50_offline_benchmarks.csv`, replacing Tables 5 and 5b with one
table: `arena, scenario, method (pelt_l2 | pelt_normal | pelt_rbf | icss | bai_perron |
causal_raw_var_cusum), localize_rate, se, penalty_or_threshold`.

**Pre-registered prediction.** H50: `pelt_normal` materially improves variance-break
localization over `pelt_l2` (from 0.00–0.20 to > 0.40 at ×3) but remains dominated by the
causal `raw_var_cusum` on the r channel at SNR 2.0.

**Decision rule.**
- H50 holds → §8.5's conclusion survives but must be narrowed from "an off-the-shelf offline
  method is not a substitute for a channel-matched statistic" to "…with a mean-shift cost,"
  and the ICSS discussion becomes the general statement.
- `pelt_normal` matches or beats the causal detector anywhere → the offline comparison is no
  longer a supporting argument for the causal framing and §8.5 must say so.

---

## §9. exp51 — Delay distributions replacing conditional-mean ARL₁

**Answers:** Major Weakness 7, and gives a sharper test of Proposition 1 than anything
currently in §4. "Mean delay conditional on detection" is upward-selected on easy paths and is
not comparable across detectors with different detection rates. Table 2 shows the pathology
directly: `raw_cusum` at SNR 0.5, variance ×3 reports mean delay 137.6 on a 0.076 detection
rate — the mean of the 7.6% easiest paths — set beside `lsc_composite`'s 25.2 at 0.992.

**Construct.** No new simulation if §0.1 is in force: this is a re-analysis of persisted
per-replicate alarm indices from exp45's runs.

Report three things per cell:

1. **Detection-rate-vs-horizon curves.** `P(detect by h)` for `h = 10, 20, …, 250` post-break
   observations, one curve per detector.
2. **Restricted mean delay (RMD).** Non-detections assigned delay = full horizon. Comparable
   across detectors by construction; this replaces the ARL₁ column in Table 2.
3. **Kaplan–Meier median delay** where censoring exceeds 50%, with the KM curve.

**The scientific payoff, not just a metric fix.** Fast-or-never makes a *shape* prediction the
current metrics cannot test: the innovation CUSUM's detection-vs-horizon curve should rise
steeply during the adaptation transient and then flatten, while the raw CUSUM's should keep
rising through the horizon (Proposition 2's positive-drift random walk detects with
probability → 1). Plot both on one axis. This is a direct visual test of Proposition 1 and it
belongs in the paper as a figure — the trichotomy currently has no figure at all.

**Pre-registered prediction.** H51: at 3σ, the innovation CUSUM's curve is concave and gains
< 0.05 detection between h = 60 and h = 250, while `raw_cusum` gains > 0.15 over the same
interval.

**Decision rule.** If the innovation curve is still rising materially at h = 250, "fast or
never" is false at this operating point regardless of what μ∞ says, and §4's framing must
change. Combined with exp52, this is the empirical half of the Proposition 1 audit.

---

## §10. Tier 2 — cheap, and each closes one specific comment

### exp52 — Numerical evaluation of the Proposition 1(b) bound *(run this first; it takes minutes)*

**Answers:** Major Weakness 1; Question 2. §4 offers "the Proposition 1 bound is never
violated" as verification. A bound that exceeds 1 cannot be violated, and on reconstruction it
appears to exceed 1 at exactly the flagship cells.

Pure computation, no simulation. For each (φ, SNR, δ) cell: solve the Riccati fixed point for
P, compute `K = P/(P+r)`, `F = P+r`, `ρ = φ(1−K)`, and
`μ∞ = δ(1−φ)/((1−φ(1−K))√F)`; take `h` from the *actual calibrated threshold* of the
innovation CUSUM in that arena; evaluate `(L+1)·exp(−2(k−μ∞)(h−g))` at `g = 0`, `L = 250`
(post-break horizon) and `L = 375` (full monitored window).

Output `paper_assets/exp52_prop1_bound.csv`:
`phi, snr, delta_sigma_ref, K, F, mu_inf, k, h, L, bound_value, bound_vacuous (bound ≥ 1),
observed_detect_est, observed_detect_known`.

Report `h` per detector per arena in the paper — it is the constant both propositions turn on
and it currently appears only incidentally (27.49/103.19/213.89 in exp38, 35.28/45.49 in exp22,
275/1829 in the φ-sweep).

**Decision rule.** If `bound_vacuous` is true at any 3σ cell: (i) §4 must stop citing
"the bound is never violated" as verification there; (ii) the abstract's "provably fast or
never" becomes "fast or never in the regime where the bound binds (δ ≤ 1σ at φ = 0.95)";
(iii) the exp10 four-corner result (0.554 / 0.636 / 0.970 / 0.990) moves out of the
"Assumptions and estimation error" paragraph into its own table, with the implication drawn
explicitly — that under the proposition's own known-parameter assumption the flagship cell
detects at 0.970 despite μ∞ < k.

### exp53 — Extend exp22's threshold/argmax diagnostic to all 12 cells

The +28.9% ARIMA-composite threshold inflation and the argmax attribution are reported for one
cell (r ×1.5, SNR 0.1). Run the same reconstruction across the full grid and report
`threshold_kalman, threshold_arima, pct_inflation` plus the argmax-feature distribution per
cell. Cheap, and §5 leans on the mechanism throughout.

### exp54 — Reduced composite on raw Y (completes the 2×2's missing cell)

The 2×2's `raw Y × composite` cell is marked "not run — raw has no filtered state." Four of the
six filtered-state features have a raw-Y analog computed on a trailing moving average of Y:
`level_change`, `slope`, `acceleration`, `instability`. Build a reduced 9-feature composite
(5 innovation-free variants + these 4) on raw Y and report it. It does not answer the
state-vs-whitening question, but it does separate "composite dilution" from "information set"
on a third information set, and it removes a visible hole from the paper's cleanest table.

### exp55 — Uniform per-window model-fit gating on all four real-data series

§9 applies the φ-clipping/model-fit gate to UNRATE only — which is the series with the
strongest nominal association. Gating only the strongest result is itself a selection. Run, for
every monitoring window of all four series: Ljung-Box p on filtered innovations (lags 6 and
12), a normality test, the unconstrained φ̂ and a clip indicator, the Kalman gain, and the
bootstrap null's own goodness of fit. Then re-run every permutation test in Tables 6 and 7
gated on well-fit windows, excluding failed windows from **both** the hit count and the
resampling universe, exactly as exp17 does.

Also report, per series, what fraction of monitored months sit in windows where the AR(1) null
is rejected — this bounds how much of the "5% FAR" label on real-data alarms is actually
verified.

---

## §11. Multiplicity ledger (maintain as experiments land)

| Family | Current size | After this package |
|---|---|---|
| §9 real-data association tests (Table 6 + Table 7, GS10 excluded) | 34 | 34 + *k* (exp47) + regated Table 6/7 (exp55 — replacements, not additions) |
| Simulation-side follow-ups | not corrected (Appendix A rationale) | unchanged; exp44–46, 48–54 join under the same rationale |

Fix *k* in exp47's pre-registration entry. State the adjusted Bonferroni threshold (α/(34+k))
and the BH ranks before running. exp55's regated tests **replace** their ungated counterparts
in the family rather than adding to it — say so explicitly, or the family will be
double-counted.

---

## §12. CHANGELOG pre-registration template

Log one entry per experiment before its first run:

```markdown
### YYYY-MM-DD — exp<NN> pre-registration (registered before running)

**Question.** <one sentence>
**Review point answered.** <Major/Minor/Specific N>
**Construct.** <module.function; grid; n_reps; seed blocks; FAR target>
**Prediction (H<NN>).** <falsifiable, with numbers>
**Decision rule.**
  - Outcome A (<condition>) → <what changes in the manuscript>
  - Outcome B (<condition>) → <what changes>
  - Outcome C (<condition>) → <what changes>
**Multiplicity.** <joins §9 family / covered by Appendix A rationale>
**Oracle status.** <causal | retrospective | known-parameter oracle>
```

and a same-day outcome entry recording CONFIRMED / FALSIFIED / MIXED against the stated
prediction, with the number that decided it.

---

## §13. What this package does and does not buy

**Closes outright:** Major 2 (exp44), Major 3 (exp46), Major 7 (exp51), Minor 1 (exp45),
Specific Comments 11, 12, 17 (exp48, exp50, exp49).

**Decides rather than closes:** Major 1 (exp52 + exp51 determine whether Proposition 1 must be
restated — it may well have to be), Major 4 (exp47 has three substantively different landing
sites, two of which require withdrawing a claim).

**Untouched by this package:** Major 5 (§9's null result and the binomial-equivalent
permutation null — a rewrite, not an experiment), Major 6 (length and MDPI format — an
editorial pass), and Major 6's cross-rung pairing question, which is answered by inspecting
whether the composite's evaluation seeds match grid_v4/v5's rather than by running anything.
Check that one before running anything else; if the answer is "no," exp45 is the moment to fix
it, since the grids are being re-run anyway.
