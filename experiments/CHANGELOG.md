# Experiment changelog (SPEC §11)

All post-hoc changes to experiment grids, with rationale. Entries are
append-only.

## 2026-07-10 — exp01: primary arena changed from local-level to AR(1) state

**What changed.** The first (30-rep smoke) version of exp01 used the
local-level DGP (random-walk latent state) as the only arena. After
seeing smoke results, the primary arena was changed to a mean-reverting
AR(1) latent state (phi=0.95, SNR 0.5); local-level was kept as a
secondary, documented hard case.

**Why (and why this is not metric shopping).** The smoke run showed
detect rates ≈ FAR for *every* method on local-level level breaks,
including at 3σ. Diagnosis: under a random-walk state model, a level
jump is statistically indistinguishable from an ordinary random-walk
innovation beyond the first observation — the well-specified Kalman
filter absorbs the break within a few steps, so no causal method has
sustained power; likewise raw-Y CUSUM is undefined-in-spirit on a
nonstationary path (its calibrated threshold was ~2300). The arena
cannot rank methods because the detection problem is nearly
unidentifiable there. The AR(1) arena makes the break identifiable
(persistent shift relative to mean-reverting dynamics) and is *more*
favorable to the raw-Y CUSUM benchmark (Y is stationary pre-break), so
the change strengthens rather than weakens the comparison. Local-level
results are still computed and reported.

**Also changed with rationale.**
- plain_hmm benchmark score changed from P(non-dominant regime) to its
  log-odds: the raw probability saturates at 1.0, so no threshold could
  achieve the 5% FAR target (the calibrated threshold hit the maximum
  of the score's support). Monotone-equivalent otherwise.
- Smoke scale (30–40 reps) results were used ONLY for these design
  decisions; all reported numbers come from fresh 500-rep runs with the
  seed layout documented in the experiment header.

## 2026-07-10 — exp01 v2: state-baseline CUSUM feature added after seeing v1 results

**What the v1 500-rep run showed.** At calibrated FAR, raw-Y CUSUM beat
the LSC innovation CUSUM on detect rate for level breaks in the AR(1)
arena (3σ: 0.98 vs 0.67), while LSC won on speed-when-detected (median
delay 23 vs 98) and on variance breaks. Mechanism identified: the
filter adapts to a persistent state shift, so innovations retain only
(1−φ)·δ ≈ 0.11σ of signal per step — below the CUSUM drift allowance
k=0.5 — and the statistic stops accumulating ("fast or never"). The raw
CUSUM tests against a fixed training baseline and accumulates the full
shift indefinitely ("slow but sure"). v1 results are preserved in
`paper_assets/exp01_v1_results.*`.

**Change.** Added two features to the diagnostics layer and the
composite: (a) `state_shift_pressure` — Page CUSUM of the filtered
state standardized by its training-prefix moments, i.e., the direct
latent-space counterpart of the raw-Y benchmark (full shift per step,
observation noise filtered out — a priori higher SNR than raw CUSUM);
(b) `variance_pressure` — CUSUM of eₜ²−1, which existed in features.py
but had been left out of the composite by oversight. Also added a
standalone `lsc_state_cusum` detector for a clean single-statistic
head-to-head against `raw_cusum`, and the ARIMA+CUSUM benchmark.

**Integrity note.** This is a method addition motivated by an
understood mechanism, not threshold tuning: all detectors, including
the new ones, are calibrated by the same null-quantile procedure at the
same 5% FAR and evaluated on the same fresh draws. The v1-vs-v2
comparison is itself reported.

## 2026-07-10 — exp02 added: SNR sweep (hypothesis stated before running)

exp01's AR(1) arena had spec-SNR (stationary state var / obs var) ≈ 5.1
— nearly noiseless observations, the regime where latent filtering buys
the least; there the latent state CUSUM only matched raw-Y CUSUM.
Mechanism-based hypothesis, registered here BEFORE exp02 was run: at
low SNR (0.1, 0.5) the filter removes a growing share of observation
noise, so lsc_state_cusum should beat raw_cusum on detect rate and
delay at matched FAR, with the gap widening as SNR falls. Break sizes
are in stationary-state-sd units so difficulty is comparable across
SNR. Same harness/seeds/reps as exp01.

## 2026-07-10 — exp02 outcome recorded; exp03 added (hypothesis stated before running)

**exp02 outcome: hypothesis falsified.** raw_cusum has the best detect
rate on level breaks at every SNR in {0.1, 0.5, 2.0} (0.97–0.99 at 3σ);
lsc_state_cusum never overtakes it (0.19→0.94 as SNR rises). The latent
innovation CUSUM is consistently the fastest conditional on detection
(median delay 24–53 vs raw's 58–91 at 3σ). Recorded in FINDINGS.md; no
grid changes made to chase a win on level breaks.

**exp03 hypothesis (registered now).** For a pure dynamics change —
AR(1) persistence jumps 0.95→0.995 or drops 0.95→0.80 at t=250, with
state-noise variance rescaled so the stationary mean AND variance of Y
are unchanged — Y-space level detectors (raw_cusum) should be blind
(detect ≈ FAR), ARIMA residual CUSUM partially sighted (its fixed AR
fit mispredicts post-break), and the LSC diagnostics layer (persistence,
instability, innovation-based features on the filtered path) should
detect well above FAR. New DGP break kind 'persistence' added with
ground-truth tests (marginals preserved, autocorrelation shifts).

## 2026-07-10 — exp03 outcome recorded

Both registered predictions were wrong in instructive ways. (1) No
method — LSC or benchmark — detects a persistence change above FAR at
spec-SNR 0.5 (up: LSC 0.03–0.07, ARIMA 0.05; down: everyone 0.01–0.03).
(2) raw_cusum was NOT blind to persistence-up (0.16): freezing the
dynamics pins the state at its break-time value, which conditionally
resembles a small never-reverting level shift. (3) persistence-down
actively SUPPRESSES every excursion-based statistic below its null
level (verified: post-break filtered-state sd 0.443 vs null 0.478;
innovation e² rises only to 1.126, below the variance-CUSUM allowance
k=0.25; Y lag-1 autocorrelation moves just 0.317→0.267). Detecting
"the series got quieter/less persistent" requires two-sided dynamics
statistics none of the tested detectors possess — logged as a v2
method direction, not patched post hoc.

## 2026-07-10 — quietness/dynamics features added; exp03b (hypothesis first)

Motivated by exp03's suppression mechanism (not by tuning on its eval
draws): added `variance_quiet` (one-sided CUSUM of 1−e², k=0.05),
`variance_pressure_slow` (k=0.05 variant for small persistent variance
rises like exp03's e²≈1.13), and `innovation_ac` (rolling lag-1
autocorrelation of innovations — white under a correct filter, serially
correlated under any dynamics change; two-sided in the composite).
Composite now has 11 features. Hypothesis for exp03b, registered before
running: the upgraded composite detects persistence breaks above FAR in
both directions (up via variance_quiet + innovation_ac, down via
variance_pressure_slow + innovation_ac), at some breadth-tax cost to
its level-break power (to be re-measured in the M5 grid).

## 2026-07-10 — composite standardization fixed: per-time-point null scales

exp03b (preserved as exp03b_pooledscale_*) showed the new quietness
features changed nothing: composite threshold was bit-identical to v1.
Diagnosis (per-feature z inspection on null vs break draws): feature
scales were pooled over ALL time points, but CUSUM-type features grow
within a path, so their late-time null values dominated the pooled
scale; the calibrated composite threshold sat at ~25 while stationary
features max out near z≈3 under breaks — the composite was blunted for
every feature, in every experiment so far. Fix: standardize each
feature at each time t by the median/IQR of the null-replication
distribution AT THAT t (self-normalization), with a floor at 10% of the
global scale. This is a design-flaw repair to the standardization, not
threshold tuning: calibration still sets the final threshold on the
same nulls at the same FAR. exp03b rerun after the fix; the M5 grid
uses the fixed composite throughout.

## 2026-07-10 — exp03b (rerun with fixed standardization) outcome

Quietness hypothesis falsified again: persistence_up 0.04,
persistence_down 0.01 — even purpose-built two-sided dynamics features
cannot see a φ 0.95→0.995/0.80 change at spec-SNR 0.5 with 250
post-break observations; the underlying signals (innovation lag-1
autocorrelation ±0.02, e² +0.13) are below any calibrated threshold's
reach. Conclusion stands on information grounds, not implementation.
The new features and per-t standardization are retained (principled,
FAR-clean: 4.8%); composite on level 3σ: 0.53 (breadth tax vs 0.75
single-feature state CUSUM), variance ×3: 0.99 / delay 26.

## 2026-07-11 — v2 robustness round registered before running: T sweep, misspecification arenas, multi-break F1

Three extensions to harden the grid_v1 conclusions, hypotheses stated
here BEFORE any of them is run. Same harness, seed layout, and 5% FAR
target throughout; n_reps = 500.

**grid_v2_T (T ∈ {200, 2000}, SNR 0.5, train_frac 0.25).** Hypotheses:
(a) the composite's subtle-variance edge (×1.5) GROWS with T — variance
evidence accumulates in e² at a constant rate, so detection should be
near-ceiling at T=2000 and clearly degraded at T=200 (only 75
post-break obs, 50 training obs); (b) raw CUSUM's level-break dominance
is unchanged at both T; (c) persistence_down (composite quietness
features) becomes detectable above FAR at T=2000 even at SNR 0.5 — the
exp03b failure was information-budget-limited, not mechanism-limited;
(d) at T=200 the composite is hurt MORE than single-statistic methods
(per-t null scales estimated from 50-obs training and 50 scale reps are
noisier, and the breadth tax bites hardest when evidence is scarce).

**grid_v2_misspec (T=500, SNR 0.5; arenas: t₅ observation noise;
nonlinear tanh state drift, drift_coef 0.1).** Calibration uses the
matched (equally misspecified-for-the-model) null, so empirical FAR
should stay ≈5% by construction for all methods — the question is
power. Hypotheses: (a) under t₅ noise every method loses power on level
breaks (heavier-tailed null maxima push thresholds up), but the
RANKING is preserved (raw CUSUM first on level, composite first on
variance ×1.5, which should survive well above FAR since e²-based
features respond to scale, not shape); (b) under nonlinear drift the
Kalman-based methods (state CUSUM, innovation CUSUM, composite) degrade
relative to raw CUSUM on level breaks because the fitted linear AR(1)
is wrong, while variance detection is roughly unaffected.

**exp04_multibreak (T=500, SNR 0.5; two breaks per path at fracs
0.4/0.7; event-level F1 with ±window matching).** New protocol code:
detectors gain a re-arm rule (after an alarm, re-arm once the score
drains below rearm_frac·threshold and a refractory has passed) applied
IDENTICALLY to every method; alarms and breaks are matched one-to-one
greedily within a 100-obs window; precision/recall/F1 per replication.
Hypotheses: (a) raw CUSUM has high recall on the first level break but
LOW recall on the second event because its statistic tests against the
fixed training baseline and stays saturated after break 1 (it never
drains → cannot re-arm); the filter-based LSC detectors adapt and
re-arm, so they win on the second event; (b) on level→variance paths
only the composite detects both events; (c) re-arming costs precision
for everyone (extra alarms), quantified at matched FAR.

## 2026-07-11 — v2 robustness round outcomes recorded

All three runs completed at 500 reps; full numbers in FINDINGS.md. No
grid or method changes were made after seeing results.

**grid_v2_T:** (a) confirmed — composite variance ×1.5 goes 0.11 →
0.87 → 0.99 across T ∈ {200, 500, 2000}; (b) held at T=200 (raw 0.80
vs composite 0.57 on level 3σ) and vacuous at T=2000 (all methods
0.98–1.00); (c) partially confirmed — persistence_down at T=2000
reaches 0.17 vs 4.6% FAR, above FAR but weak; (d) confirmed — at T=200
the composite is the most degraded method AND the hottest calibration
(9.4% empirical FAR from heavy-tail threshold noise at short T).

**grid_v2_misspec:** (a) partially confirmed — under t₅ noise rankings
are preserved and raw_cusum is unhurt (0.99 level 3σ), but the
composite's variance ×1.5 edge collapses to 0.16 (×3 intact, 0.97);
the pre-registered guess that the subtle-variance case "should survive
well above FAR" was too optimistic — heavy tails inflate the null e²
CUSUM directly. (b) wrong in an instructive way — nonlinear drift does
not merely degrade the Kalman methods relative to raw CUSUM: it
destroys level detection for EVERY method (raw 0.15 at 3σ) by
inflating null thresholds via the state's own bimodal hopping, while
composite variance detection is fully intact (0.97 at ×1.5, better
than the Gaussian arena). Tail-robust innovation features logged as
the highest-value v2 method item; no post-hoc patching done.

**exp04:** (a) half-confirmed — raw CUSUM is structurally one-shot
(second-event recall 0.00 everywhere) as predicted, but the LSC
detectors also rarely re-arm in time at 150-obs spacing (level→level
second-event recall ≤ 0.05): first-alarm delay plus slow CUSUM drain
exceeds the event gap. (b) confirmed — level→variance is caught by the
composite alone (second-event recall 0.60, F1 0.63). (c) wrong in a
good way — re-arming costs almost nothing (≤1.2% of null paths give a
second alarm); the binding constraint is saturation, not chatter.
Bounded-memory (windowed) statistics logged as the v2 fix.

## 2026-07-11 — registered before running: robust variance features (exp05) and theory verification (exp06)

**exp05 — Huberized second-moment features.** Motivated by the
grid_v2_misspec t₅ failure (variance ×1.5: 0.87 Gaussian → 0.16 t₅),
mechanism understood: e² is heavy-tailed under t₅, so the null CUSUM of
(e²−1) has occasional huge jumps that inflate the calibrated threshold.
Fix: clip. New features `variance_pressure_robust` /
`variance_quiet_robust` — innovations rescaled by their training-prefix
robust scale (1.4826·MAD), clipped at c=2.5, squared, centered and
standardized by the TRAINING-PREFIX mean/sd of the clipped square
(causal), then one-sided CUSUMs (k=0.05 each way). Clipping bounds the
summand, so the null max is thin-tailed regardless of the innovation
distribution; training-moment centering adapts to whatever null
distribution the arena has. The existing composite is FROZEN as the
11-feature set (new `COMPOSITE_V1` include-list, default of
make_composite_detector — all previous results stay bit-reproducible);
a new method `lsc_composite_robust` swaps the three e²-based features
for the two robust ones. Hypotheses, registered before running
grid_v3_robust.yaml (Gaussian + t₅ arenas × {level_3s, variance_x1.5,
variance_x3, persistence_down} × {composite, composite_robust,
raw_cusum}): (a) under t₅, composite_robust recovers variance ×1.5 to
≥ 0.5 detect (from composite's 0.16); (b) under Gaussian noise the
robustness tax on variance scenarios is ≤ ~10pp; (c) level and
persistence rows are roughly unchanged (those features are shared).

**exp06 — fast-or-never made precise (theory + numerical check).**
Claim to verify: for a steady-state Kalman filter (gain K, innovation
sd √F) on the AR(1)+noise model, a state level shift δ at t₀ adds a
deterministic mean path to the standardized innovations,
μ_t = μ_∞ + (δ−μ_∞·√F)·ρ^{t−t₀}/√F with ρ = φ(1−K), decaying
geometrically to μ_∞ = δ(1−φ) / ((1−φ(1−K))·√F). If μ_∞ < k (the CUSUM
drift allowance), the post-transient CUSUM has negative drift and a
finite-horizon Lundberg/union bound gives P(alarm in remaining L obs)
≤ L·exp(−2(k−μ_∞)h) — "never"; detection can only come from the
transient — "fast". For raw CUSUM the standardized shift Δ = δ/σ_Y
persists forever, so if Δ > k detection is certain with Wald delay
≈ h/(Δ−k). Numerical predictions computed from the formulas BEFORE
running exp06 (SNR 0.5 arena, k=0.5): K=0.165, √F=1.094, μ_∞(3σ) =
0.469 — a knife-edge 0.031 below k, consistent with the observed
partial detect rates 0.55–0.67; μ_∞(1σ) = 0.156 → bound ≈ 7e−5 per 250
obs at h≈22 → never (observed ≈ FAR); raw Wald delay at 3σ with
h≈103 ≈ 84 obs (observed median 82–91). exp06 checks (i) the mean
path against full-filter MC, (ii) the reduced CUSUM (z+μ_t) against
full MC detect rates over an h grid vs the bound, (iii) the μ_∞ table
against grid_v1 innovation-CUSUM detect rates. Derivations in
experiments/THEORY.md.

## 2026-07-11 — exp05 outcome: FALSIFIED; exp05b (exceedance CUSUM) registered before running

**exp05 outcome.** The clipped features are worse everywhere, not
better: under t₅, variance ×1.5 went 0.16 (composite) → 0.03
(composite_robust) and ×3 0.97 → 0.46; even under Gaussian noise the
robust composite collapsed on its target case (×1.5: 0.87 → 0.06; ×3
0.99 → 0.82 with median delay 26 → 173). All three pre-registered
hypotheses wrong. Mechanism (understood after the fact): a variance
rise manifests almost entirely in the TAIL of e² — the clip at
2.5·MAD removes the same observations that carry the signal, so
clipping buys a thin-tailed null by destroying the alternative. The
clip-based features remain in the codebase as a documented negative
result but in no recommended composite.

**exp05b — exceedance-rate features, registered now.** The right
robust statistic must have a bounded null summand while keeping tail
SENSITIVITY: use the exceedance INDICATOR, not the exceedance
magnitude. New features `tail_exceedance` / `tail_shortfall`:
one-sided CUSUMs of (1{|e_t| > q̂90} − p̂ − k) and (p̂ − 1{|e_t| > q̂90}
− k), with q̂90 the training-prefix 90% quantile of |e|, p̂ the
training exceedance rate (≈0.10 by construction), k = 0.02. The
summand is bounded in [−1, 1] under ANY innovation distribution
(thin-tailed null maxima), while a ×1.5 scale rise moves the
exceedance probability 0.10 → ≈0.27 (Gaussian) / ≈0.24 (t₅) — drift
≈ 0.12–0.15 per step against summand sd ≈ 0.3–0.35.
`COMPOSITE_ROBUST2` = COMPOSITE_V1 with the three e²-based variance
features replaced by the two exceedance features. Hypotheses for
grid_v3b (same arenas/scenarios/methods pattern as exp05): (a) under
t₅, composite_robust2 detects variance ×1.5 ≥ 0.5 (composite: 0.16);
(b) under Gaussian noise its variance ×1.5 power is within ~15pp of
the composite's 0.87; (c) level/persistence rows unchanged (shared
features).

## 2026-07-11 — exp05b outcome: falsified AT THE COMPOSITE LAYER; exp05c registered before running

**exp05b outcome.** composite_robust2 sat at FAR on variance ×1.5 in
both arenas (0.06 Gaussian / 0.03 t₅). Diagnosis (on calibration-range
and 400000+ seeds, NOT evaluation seeds): the exceedance FEATURE
separates almost perfectly — its raw CUSUM max under a ×1.5 break
exceeds the null 95th percentile on 93–100% of paths, in both arenas —
but the composite kills it: with allowance k=0.02 the null exceedance
CUSUM wanders (drift −0.02), inflating its per-time-point null IQR, so
its standardized break z tops out around ~15 while the composite's
calibrated threshold (~28) is set by break_pressure/instability null
tails. A max-over-features composite rewards break-to-null-IQR RATIO,
and the bounded-increment exceedance CUSUM cannot reach the ratios the
unbounded e² CUSUM achieves under Gaussian nulls.

**exp05c — registered now.** Two changes, both disclosed: (i) the
allowance for `tail_exceedance` is raised to k=0.05, chosen by a
separation diagnostic (null95 vs break median of the raw feature) run
ONLY on non-evaluation seeds (nulls 100000+, breaks 400000+; k grid
{0.02, 0.05, 0.10, 0.15}); `tail_shortfall` stays at k=0.02 (the
quieting drift is small, ~0.056/step at ×⅔ scale, and cannot afford a
large allowance). (ii) The exceedance statistic is ALSO exposed as a
standalone calibrated detector `lsc_tail_cusum` (score = max of the
two arms), mirroring the exp01-v2 precedent of lsc_state_cusum: the
statistic is strong but the composite's standardization layer
compresses it, so it gets a clean single-statistic head-to-head at
matched FAR. Scenario set gains variance_x0.67 (subtle quieting, the
shortfall arm's target). Hypotheses for grid_v3c: (a) standalone
lsc_tail_cusum detects variance ×1.5 at ≥ 0.9 Gaussian / ≥ 0.8 t₅ —
i.e. the t₅ repair the clip approach failed to deliver; (b)
composite_robust2 (with k=0.05) improves over exp05b but may remain
well below the standalone (z-compression is structural); (c)
variance_x0.67 quieting is detected above FAR by lsc_tail_cusum
(predict ≥ 0.25) in both arenas; (d) raw_cusum stays at FAR on all
variance rows.

## 2026-07-11 — exp05c outcome recorded

(a) Substantially confirmed, slightly under the registered bars:
lsc_tail_cusum detects variance ×1.5 at 0.87 Gaussian (bar: ≥0.9) and
0.75 t₅ (bar: ≥0.8) — the t₅ repair is real (composite: 0.16) but both
numbers miss the registered thresholds by 3–5pp, reported as such.
(b) Confirmed: composite_robust2 rose to 0.58 Gaussian / 0.21 t₅ —
better than exp05b's 0.06/0.03, still far below the standalone
(z-compression is structural, as registered). (c) Confirmed: subtle
quieting ×0.67 detected at 0.41 / 0.33 (every other method 0.02–0.04)
— the first successful quieting detection in the project. (d)
Confirmed: raw_cusum at FAR on every variance row. FAR calibration
3.6–6.2%. lsc_tail_cusum also incidentally fires on level 3σ breaks at
0.30–0.37 (transient innovation exceedances) and hits 0.99–1.00 on ×3
with delay ~37 under BOTH distributions. Recommendation recorded in
FINDINGS: lsc_tail_cusum is the second-moment detector of choice under
distributional uncertainty; COMPOSITE_V1 remains the breadth
instrument; the clip-based features (exp05) and the composite-embedded
exceedance variant (exp05b) are documented negative results. No
further tuning after these results.

## 2026-07-11 — M0 (varbench addendum): claim-adoption decision rule PRE-REGISTERED before any grid_v4 cell runs

Registered per `SPEC_addendum_varbench.md` §1, before implementing or
running the new variance benchmarks. The paper will adopt whichever
claim the results select; no post-hoc reinterpretation.

**New detectors (whitening ladder, M1).** `raw_var_cusum` — Page CUSUM
of z_t²−1 with z_t = (Y_t − ȳ_train)/σ̂_train frozen from the training
prefix; up-arm allowances k = 0.25 and k = 0.05 (mirroring the latent
variance_pressure / variance_pressure_slow features exactly), down-arm
(quieting) CUSUM of 1−z_t² with k = 0.05; score = max over the three
arms; no per-time-point standardization (standalone detector, same
treatment as lsc_tail_cusum). `arima_var_cusum` — the identical
statistic on the standardized one-step residuals of the existing ARIMA
benchmark's training-prefix-fitted, frozen model. Ladder: raw →
ARIMA-whitened → Kalman-whitened (existing e²-based CUSUMs), same
statistic, same allowances, same calibration routine, three
information sets.

**Decision rule.** Let D_raw = raw_var_cusum detection rate at
variance ×1.5, T = 500, per SNR ∈ {0.1, 0.5, 2.0}; D_comp = the
composite's published 0.82 / 0.87 / 0.91 (grid_v1, identical seeds);
FAR target 5%.

- **Outcome A (strong claim):** D_raw within 5 pp of FAR at every SNR
  → abstract/intro upgrade to "raw-data detectors, including a
  variance CUSUM given identical calibration, sit at chance."
- **Outcome B (prewhitening claim):** D_raw within 10 pp of D_comp at
  every SNR → reframe: the advantage is *prewhitening under
  autocorrelation*, not latency per se; §5 and §11 rewritten
  accordingly (the fast-or-never side is untouched). Honest-outcome
  clause: if the latent state layer is unnecessary for second moments,
  the paper says so plainly — "for whitening, not for state
  estimation" — and does not soften it.
- **Outcome C (mixed):** anything else → report the full ladder, keep
  the scoped language, add a paragraph explaining the SNR-dependence.

All three outcomes are publishable; which fired will be logged here
with the numbers.

**Predictions registered alongside (not gating).** (i) Under t₅ noise
raw_var_cusum's ×1.5 power should collapse like the composite's did
(0.87 → 0.16) — z² has the same tail sensitivity; if so, the
exceedance repair story extends to the raw side. (ii) The variance
detectors should NOT detect level shifts above FAR (disjoint-channels
table). (iii) Real-data reruns (M3) may weaken the real-data
uniqueness claim — that is the point of running them; if raw_var_cusum
catches the same crises, §10's claim becomes about the
simulation-calibrated subtlety threshold (×1.5 invisibility), not
real-data uniqueness.

## 2026-07-11 — real-data extension registered before running (m6x)

All real-data results remain ILLUSTRATIVE (SPEC §4.5/§8); this entry
registers the design, not power hypotheses. Components:

1. **Data snapshots.** Date-stamped CSVs committed under `data/`;
   loaders prefer the snapshot (live download only on request) so
   published numbers survive FRED revisions.
2. **Generalized engine** (`experiments/real_data.py`), replacing
   nothing — `m6_fred.py` is kept untouched. Detector set now includes
   `lsc_tail_cusum`; composite alarms are ATTRIBUTED (which feature's
   z crossed at the alarm), tail alarms attributed to the up/down arm.
3. **New series, chosen for the method's signature cases before
   looking at their results:** (a) quarterly real GDP growth (GDPC1) —
   the 1984:Q1 Great Moderation is the canonical QUIETING event
   (McConnell–Perez-Quiros 2000); (b) monthly 10-year Treasury yield
   changes (GS10) — the 1979–82 Volcker episode is the canonical
   volatility-UP regime, the post-2008 ZLB a quieting. Reference
   events fixed in the script header.
4. **Sensitivity variants** for INDPRO: (train, monitor) = (180, 36)
   and FAR = 10%, alarms compared to the baseline set.
5. **Evaluation** (`real_data_eval.py`): per series/method, hits
   within 12 months (4 quarters) after reference events, expected
   vs observed non-event alarms at the calibrated FAR, and a
   permutation p-value (alarm months resampled uniformly over
   monitored months, 20k draws) for the event-association.
6. **Real-time vintage check** (`realtime_check.py`, ALFRED
   `alfredgraph.csv?vintage_date=`): month-by-month decisions for the
   GFC (2008-01..2009-06) and COVID (2020-01..2020-12) episodes,
   training on the 120 months ending at the (later-declared) NBER
   peak, recalibrated per vintage. The revised-data timing claim
   ("composite alarmed 2008-09") counts as robust only if the
   real-time alarm month matches within ±1 month; otherwise the paper
   claim is downgraded to revised-data-only.
