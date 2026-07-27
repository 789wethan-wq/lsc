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

## 2026-07-12 — varbench outcome: **Outcome C fired** (M4 resolution)

The pre-registered decision rule (M0 entry above) is now resolved
against the completed grid_v4_varbench run (500 reps, identical seed
blocks, empirical FARs 4.2–6.8%).

**Numbers (raw_var_cusum, variance ×1.5, T = 500).** D_raw = 0.996 /
0.560 / 0.102 at SNR 0.1 / 0.5 / 2.0; empirical FAR ≈ 6%; the composite
D_comp = 0.82 / 0.87 / 0.91.

- **Outcome A** (D_raw within 5 pp of FAR at *every* SNR) — FALSIFIED:
  0.996 and 0.560 are far above FAR (only the SNR-2.0 cell, 0.102 vs
  6.0%, is near chance).
- **Outcome B** (D_raw within 10 pp of D_comp at *every* SNR) —
  FALSIFIED: off by 18 / 31 / 81 pp.
- **Outcome C (mixed) — FIRED.** The raw variance CUSUM is strongly
  SNR-dependent: it *beats* the composite when observation noise
  dominates (0.996 at SNR 0.1) and collapses to chance as the latent
  signal grows (0.102 at SNR 2.0). Mechanism: as SNR rises the latent
  state's variance dominates Var(Y), so a ×1.5 change in the shrinking
  noise component is masked by state-driven autocorrelation.

**Sharpening sub-finding (whitening rung).** The middle rung
`arima_var_cusum` on the same ×1.5 break is essentially flat across SNR
(0.90 / 0.94 / 0.87) and tracks the composite step for step —
prewhitening recovers the full second-moment advantage, and the latent
*state estimate* adds little the ARIMA residuals do not. Reported per
the honest-outcome clause (SPEC §8): for second moments the edge is
"whitening, not state estimation." Under t₅ the composite collapses
(×1.5: 0.16) while the raw (0.43) and ARIMA (0.74) variance rungs hold —
the plain z²/e² max-over-arms statistic keeps the tail signal the
per-time composite standardization discards.

**Real-data resolution (M3).** raw_var_cusum was added to the INDPRO /
GDP / GS10 monitoring and (because it alarmed on revised INDPRO at
2008-09) to the ALFRED vintage protocol. It catches every headline
crisis — GFC (INDPRO 2008-09, GDP 2009Q2), COVID (GDP 2020Q2), Volcker
(GS10 1980-02) — and its real-time vintage timing is *identical* to the
composite for both GFC (2008-12 vintage, data month 2008-09) and COVID
(2020-04 vintage, 2020-03). So the real-data *uniqueness* claim is
downgraded per the M0 spirit: the crises are not uniquely detected by
the latent layer. What survives is (i) the composite's clean association
profile — significant, low-stray (INDPRO p = 0.007 vs raw_var p = 0.56,
which strays on quieting alarms and washes out; GS10 raw_var 9 alarms /
8 stray) — and (ii) the simulation-calibrated ×1.5 subtlety threshold at
high SNR, where only the whitened rungs see the break. The 180-month
training window breaks *both* variance detectors (composite and raw_var
each 14 alarms), isolating the sensitivity to the second-moment
statistic rather than the composite machinery.

**Paper patch (M4).** PAPER_DRAFT.md updated: abstract and §1
contribution 2 reframed to the prewhitening finding; §5 rewritten as the
whitening ladder with the raw/ARIMA/latent table and a T-sweep note;
§8.2 gains the t₅ raw/ARIMA rungs; §9 gains raw_var_cusum rows, the
apples-to-apples vintage comparison, and the downgraded uniqueness
language; §10 recipe becomes "whiten, then run the variance CUSUM." All
editorial `[EDIT]` brackets resolved; propositions numbered; test count
77 → 84.

## 2026-07-12 — full `make all` reproduction gate + stale-artifact correction

Ran the entire deterministic suite from the committed code (`make all`,
~2 h, green, 0 errors) as the clean-clone reproducibility gate. Findings,
with the adjudication trail:

**Result.** 7 of 12 experiment groups regenerate BYTE-IDENTICAL, including
every paper-quoted source: grid_v1, grid_v2 (T + misspec), grid_v3_robust,
grid_v3c, grid_v4, exp04, exp06. grid_v4_varbench was separately confirmed
bit-identical on re-run from pinned seeds (all 60 rows, thresholds, FARs).

**5 groups had STALE committed artifacts** — parquets generated by an
earlier (pre-git) version of the composite code and never regenerated
before the initial commit bundled them with the newer code:
- exp01 / exp02 / exp03: only `lsc_composite` rows move (≤0.036); the
  single-statistic detectors and exp02's cited `lsc_state_cusum` numbers
  are unchanged.
- grid_v3b_exceedance: only `lsc_composite_robust2` moves (both arenas;
  ×1.5 SNR 0.5 0.058→0.582, ×1.5 t₅ 0.034→0.208, ×3 t₅ 0.506→0.972); the
  default `lsc_composite` rows in the same grid are byte-identical.
- m2_param_recovery: `markov_switching` rows differ at ~1e-13 (float /
  BLAS-threading epsilon, not the RNG issue; rounds identical).

**Git archaeology.** No commit in this repo altered the composite score:
`features.py` and `alarms.py` (the exceedance features and the per-t
null-standardization) are byte-identical to the initial commit; M1's only
touch to `detectors.py` ADDED the two variance-detector factories and left
`make_composite_detector` untouched. The change that moved robust2
therefore predates `git init` (the pre-git exp05 "pooled scale → per-t
standardization" fix logged in §8.4), and the stale parquet slipped in
because no test pins grid_v3b's output.

**Hand adjudication (not a re-run).** On one ar1_snr0.5 path we recomputed
the `tail_exceedance` feature's per-t null center/IQR manually and matched
the pipeline exactly (max |Δ| = 0.0), hand-standardized its score at
several t (e.g. t=400: (14.317−0.805)/1.520 = 8.89), and confirmed the
max-over-features composite equals `det(Y)` at every post-train t
(max |Δ| = 0.0), with `tail_exceedance` the dominating feature post-break
(z 2.1 → 8.9 → 19.2). Current code (0.582) is arithmetically correct; the
committed 0.058 was the pre-fix pooled-scale output.

**Actions.** (1) Committed the regenerated artifacts so `make all` is
idempotent and every number is backed by the current code. No
paper-QUOTED number changes. (2) Corrected §8.3(ii): the exceedance
indicator is *diluted, not dead*, inside the composite (0.58 / 0.21, even
edging the e²-composite under t₅); its per-t score reaches z ≈ 19, so the
old "bounded increments can't reach the ratios" mechanism was an artifact
of the pre-§8.4 pooled-scale build. The standalone recommendation (iii)
stands — now justified by FAR-budget dilution across ~10 features, not by
the composite crushing the feature.

## 2026-07-13 — M0 (R1 referee-hardening): M1/M2 decision rules PRE-REGISTERED before any R1 artifact

Registered per `SPEC_latent_state_change.md` R1 round §M0, before
generating any R1 artifact (no exp07 file, no grid_v5 config, no
grid_v5/v6 parquet exists at commit time — verified: `git status` clean,
`ls experiments/exp07* configs/grid_v5* configs/grid_v6*` empty). The
paper adopts whichever branch the results select; no post-hoc
reinterpretation.

**M1 rule — ARMA(1,1) equivalence [GATE].** Let ρ̄ = median Pearson
correlation between the ARIMA standardized-residual series and the
Kalman standardized-innovation series over the post-burn-in monitoring
region, across null (no-break) paths, at each SNR ∈ {0.1, 0.5, 2.0},
T = 500, ≥200 paths.

- **A1 — Equivalent (expected): ρ̄ ≥ 0.95.** The two rungs are the same
  filter up to estimation error. §5 is reframed as a two-rung ladder
  (raw vs. whitened) with the ARMA(1,1) reduced-form equivalence stated
  as theory and the grid reported as numerical confirmation. Practical
  recipe (§10) sharpens to "ARMA whitening suffices; the state-space
  layer is not required for second-moment monitoring."
- **A2 — Not equivalent: ρ̄ < 0.95. STOP.** For an AR(1)+white-noise
  DGP the steady-state Kalman innovations and the ARMA(1,1) innovations
  are the same linear innovations of the same Gaussian process; a large
  gap implies an estimator bug (candidate causes: ARIMA order
  misselection — especially AIC choosing a differencing order —
  non-frozen parameters, standardization mismatch, burn-in
  contamination). Diagnose the estimator before running anything else.
  Do NOT proceed to M2–M4.

Regression guard: `test_arma_kalman_equivalence` asserts ρ̄ ≥ 0.95 with
TRUE (not estimated) parameters, where agreement should be near-exact.

**M2 rule — q-break (state-innovation variance) ladder.** Let R_q =
ladder detection rates (raw / ARIMA-whitened / Kalman-whitened) for a
state-innovation-variance break, and R_r = the published
observation-noise (r) results at ×1.5, T = 500: raw 1.00 / 0.56 / 0.10,
whitened ≈ 0.90 / 0.94 / 0.87 across SNR 0.1 / 0.5 / 2.0. Convention:
the q-break scales the state-innovation **SD** by the same `vol_mult`
the r-break scales the observation-noise SD by (verified: the existing
`variance` break multiplies obs-noise std, per `obs_noise_scale_path`
and the `BreakSpec` docstring), so "×1.5" means SD×1.5 in BOTH channels
— the only setting under which the two-channel comparison is
meaningful.

- **B1 — Ordering survives:** raw remains strongly SNR-dependent and
  whitened remains approximately flat → the §5 claim generalizes; report
  both break channels; paper materially stronger.
- **B2 — Ordering inverts or flattens:** the "prewhitening beats raw"
  result is specific to white-component (r) breaks → §5's claim is
  scoped to r-breaks, and the paper MUST say so, because every
  real-data motivating event (Great Moderation, crisis volatility) is a
  q-break. §9–§10 framing revised accordingly (honest-outcome clause).
- **B3 — Mixed / SNR-dependent in a new way:** report the full
  two-channel ladder and characterize.

All M1/M2 outcomes are publishable. The fired branch is logged with
numbers when M1/M2 resolve.

## 2026-07-13 — M1 [GATE] RESOLVED: **A1 fires** — the ARIMA and Kalman rungs are the same filter

`experiments/exp07_arma_equivalence.py` (≥200 null paths, T=500, SNR ∈
{0.1, 0.5, 2.0}); theory + machine-precision cross-checks in
`experiments/THEORY.md` (§"ARMA(1,1) equivalence") and
`lsc.theory.arma11_representation`; regression guards
`test_arma11_riccati_identities` + `test_arma_kalman_equivalence`.

**Structural theory (machine precision).** The observable Y has an exact
ARMA(1,1) reduced form with AR parameter φ, MA parameter θ, innovation
variance σ_ε². Two identities hold to < 1e-12: σ_ε² = F (the Riccati
innovation variance of Proposition 1) and θ = ρ = φ(1−K) (the
innovation-mean decay rate). Hence the steady-state Kalman innovations
and the ARMA(1,1) innovations are the same linear innovations of the
same Gaussian process.

**Empirical (deliverable `paper_assets/arma_equivalence.csv`).**
- TRUE parameters: median Pearson ρ = 1.000000 at every SNR; max|Δ|
  between the (independent) statsmodels ARMA filter and the hand-written
  steady-state Kalman recursion ≈ 1e-9 (1.5e-7 at SNR 0.1). Near-exact,
  as predicted — the wedge is entirely estimation error.
- ESTIMATED parameters (ladder's real operating condition): ρ̄ = 0.9914
  (medians 0.9947 / 0.9914 / 0.9880 at SNR 0.1 / 0.5 / 2.0), ≥ 0.95 ⇒
  **A1**. Forcing the ARIMA order to the true (1,0,1) tightens this to
  ρ̄ = 0.9995 (0.9992 / 0.9995 / 0.9996), so AIC order-selection is the
  sole material wedge.

**Order-selection finding (SPEC M1 asked for this explicitly).** AIC over
the benchmark grid selects the true (1,0,1) on only 16.5% / 9% / 5.5% of
paths; it prefers (1,0,0) at SNR 0.1 (36.5%) and the differencing order
(0,1,1) at SNR 0.5 / 2.0 (63.5% / 71%). This is a near-unit-root artifact
(φ = 0.95), NOT a code bug: at φ≈1 both IMA(1,1) and AR(1) approximate
the ARMA(1,1) closely enough to keep ρ̄ ≥ 0.95, and the AIC-selected
model is a legitimate whitener. Not changing the published varbench
ARIMA rung (AIC selection is the benchmark's design and the change would
ripple into grid_v4); reported as-built, with the forced-(1,0,1) column
alongside and a scope note that mis-differencing would bite at small φ
(picked up in M3).

**Paper action (M6).** §5 reframed as a **two-rung ladder** (raw vs.
whitened); ARMA(1,1) equivalence stated as theory (Appendix B), grid as
numerical confirmation; §10 recipe sharpened to "ARMA whitening
suffices; the state-space layer is not required for second-moment
monitoring." GATE PASSED — proceeding to M2.

## 2026-07-13 — M2 RESOLVED: **B2 fires** — prewhitening's advantage is specific to observation-noise (r) breaks

`configs/grid_v5_qbreak.yaml` (state-innovation q-break, SD-scaled to
match the r-break convention; 500 reps, T=500, identical seed blocks);
new DGP kind `state_var` in `lsc/dgp/breaks.py` +
`state_noise_scale_path`, wired into AR1StateDGP and LocalLevelDGP
(tests `test_state_var_break_scales_state_innovation_sd`,
`test_state_var_null_matched_and_reproducible`). Assembler
`experiments/qbreak_ladder.py` → two-channel `ladder_table.csv`
(`break_channel` r|q). q-break FARs 3.6–6.8% (calibrated).

**The ladder ordering INVERTS across channels** (detect rate at the
discriminating ×1.5 break, T=500, SNR 0.1/0.5/2.0):

| chan | rung  | 0.1  | 0.5  | 2.0  |
|------|-------|------|------|------|
| r    | raw   | 1.00 | 0.56 | 0.10 |  (falls with SNR)
| r    | ARIMA | 0.90 | 0.94 | 0.87 |  (flat — WINS at high SNR)
| q    | raw   | 0.09 | 0.21 | 0.23 |  (RISES with SNR — WINS everywhere)
| q    | ARIMA | 0.03 | 0.10 | 0.16 |  (below raw at every SNR)

For q-breaks the raw variance CUSUM ≥ the whitened rung at every SNR
(and the composite): ×1.5 raw beats ARIMA 0.09/0.21/0.23 vs
0.03/0.10/0.16; ×3 raw 0.72/0.96/0.96 vs ARIMA 0.26/0.79/1.00 (ARIMA
only catches up at the ×3 SNR-2.0 ceiling). And the raw rung's
SNR-dependence FLIPS sign: r-break raw falls 1.00→0.10, q-break raw
rises 0.09→0.23.

**Mechanism (clean and symmetric).** An r-break lives in the WHITE
component of Y; state autocorrelation masks it in raw z², and whitening
removes exactly that autocorrelation → whitening wins, and raw fades as
SNR rises and the white component shrinks. A q-break lives in the STATE
(autocorrelated) component; it inflates the marginal variance of Y — which
dominates at high SNR — so raw z² sees it directly, while whitening
strips out the state-carried signal → raw wins, and raw strengthens as
SNR rises. A q-break also shifts the ARMA(1,1) MA parameter θ (verified:
θ 0.793→0.710 at ×1.5, SNR 0.5), changing the autocorrelation structure,
whereas an r-break changes only the marginal variance.

**Decision: B2** (pre-registered). "Prewhitening beats raw" is specific
to observation-noise breaks. **Honest-outcome clause invoked**: §5 and
§10 are scoped to r-breaks; the paper states plainly that for the
state-innovation (q) breaks that motivate the empirical section — Great
Moderation, crisis volatility are q-like shifts — a raw variance CUSUM is
at least as good as whitening at every SNR. This also *explains* the §9
real-data result (raw_var_cusum's crisis timing is identical to the
composite's): real crises are q-channel breaks, exactly where B2 says raw
matches whitened. Abstract revised (B2 changes the headline). Quieting
(×⅔ i.e. q reduced) is undetectable by every rung (≤0.07, at FAR) — a
low-q state contributes too little to Y to register its own reduction.
Under t₅ the q-break ×3 ranking holds with raw most robust (raw 0.93,
ARIMA 0.66, composite 0.48, tail 0.79). All numbers in
`paper_assets/grid_v5_qbreak_results.parquet` + `ladder_table.csv`.

## 2026-07-13 — M3: φ sweep confirms μ∞ sorts detection — with a boundary condition

`configs/grid_v6_phisweep.yaml` (level 1σ/3σ × φ∈{0.5,0.8,0.95,0.99} ×
SNR∈{0.1,0.5,2.0}, T=500, 500 reps; SNR held fixed across φ by q =
SNR·(1−φ²)); analysis `experiments/phisweep_analyze.py`. Deliverables
`paper_assets/grid_v6_phisweep_muinf.csv`, `grid_v6_muinf_scatter.png`
(headline theory-verification figure).

**μ∞ sorts detection: Spearman(μ∞, innovation-CUSUM detect) = 0.942**
across all 24 cells. Because μ∞ = δ(1−φ)/((1−φ(1−K))√F) is increasing in
(1−φ), the innovation CUSUM ESCAPES the fast-or-never regime at low φ:
3σ detection is 0.98/1.00 at φ=0.5 and 0.93/0.97 at φ=0.8 (μ∞ ≥ 0.69,
FAST) but collapses to 0.65/0.55 at φ=0.95 and 0.63/0.30 at φ=0.99
(μ∞ ≤ 0.48, "never"). Fast-regime cells (μ∞ ≥ k=0.5) detect 0.83–1.00;
never-regime cells (μ∞ < 0.5) detect 0.07–0.67. The crossover sits
between φ=0.8 and φ=0.95, exactly at the predicted k=0.5 boundary.

**Boundary condition for the paper.** Proposition 1's fast-or-never is
therefore not an unqualified claim: filtering fails for LEVEL detection
only when the state is PERSISTENT (φ ≳ 0.9) — which is the empirically
relevant case (φ=0.95 is the paper's baseline). At low persistence the
innovation CUSUM is competitive with raw. raw_cusum, by contrast, detects
3σ levels at 0.96–1.00 across ALL φ (φ-robust, slow-but-sure), so the
φ-dependence is specific to the filtered detector — a direct corroboration
of the mechanism.

**Honest caveat (the one off-trend cell).** At the near-unit-root corner
φ=0.99, SNR 0.1, 3σ, detection is 0.63 despite μ∞=0.21 (deep "never") —
above the μ∞ trend. Mechanism: at φ→1 the adaptation transient (decay
rate ρ=φ(1−K)) is so long that its accumulated mass triggers the CUSUM
during the transient even though the asymptotic drift is negligible —
i.e. the "fast" branch of fast-or-never is itself stronger when the
transient is long. So μ∞ sorts the ASYMPTOTIC regime cleanly (Spearman
0.94) but finite-sample detection at very high φ also carries a transient
contribution; reported as such, not smoothed over. Proposition 1 holds;
the caveat is that total detection = transient mass + post-transient
tail, and μ∞ governs only the second.

## 2026-07-13 — M7 PRE-REGISTERED: φ × q-break cross-grid (amplification test), before any grid_v8 cell runs

Registered before implementing `configs/grid_v8_phiqbreak.yaml` (verified:
no grid_v8 config/parquet at commit time). Extends M2 (q-break) × M3 (φ)
into a genuine cross: hold the shock variance q and obs variance r FIXED
and sweep φ, so the state's stationary variance q/(1−φ²) — and hence the
signal-to-noise ratio SNR(φ) = (q/r)/(1−φ²) — rises as the **1/(1−φ²)
amplification factor**. Anchor q = 0.04875, r = 1.0 (so φ = 0.95 gives
SNR 0.5, matching the body arena); φ ∈ {0.1, 0.5, 0.7, 0.85, 0.95, 0.99}
gives induced SNR {0.049, 0.065, 0.096, 0.176, 0.5, 2.45}. q-break ×1.5
and ×3; rungs raw_var_cusum / arima_var_cusum / lsc_composite; 500 reps,
T = 500, identical seed blocks.

**Pre-registered prediction (falsifiable).** The raw rung's *advantage*
on q-breaks — Δ(φ) = detect(raw_var) − detect(arima_var) (and vs
composite) — is (i) monotonically increasing in φ and (ii) → 0 as φ → 0,
(iii) tracking the amplification 1/(1−φ²) (equivalently the induced SNR).
Mechanism: at φ → 0 the observable is white (AR(0)), so whitening is a
no-op and the q-break barely moves Y's variance → raw ≈ whitened ≈ FAR,
Δ ≈ 0; as φ → 1 the 1/(1−φ²) amplification inflates the state's share of
Y's variance, which a raw z² statistic sees directly while whitening
strips it out → Δ large. Secondary consistency check: Δ at each cell
should match the M2 SNR-sweep (grid_v5, fixed φ = 0.95) at the *same
induced SNR* — i.e. the effect operates through amplified SNR, not φ per
se. **Falsifiers:** Δ(φ) non-monotone; Δ(0.1) materially > 0; Δ present
without autocorrelation; Δ inconsistent with the M2 SNR-sweep at matched
SNR. Outcome logged with numbers when resolved; all outcomes publishable.

## 2026-07-13 — M7 RESOLVED: prediction CONFIRMED IN PART, FALSIFIED IN PART (honest mixed)

`configs/grid_v8_phiqbreak.yaml` (fixed q=0.04875, r=1; φ ∈ {0.1, 0.5,
0.7, 0.85, 0.95, 0.99}; induced SNR 0.049→2.45); analysis
`experiments/phiqbreak_analyze.py` → `grid_v8_phiqbreak_summary.csv`,
`grid_v8_phiq_amplification.png`. Raw advantage Δ(φ) = detect(raw_var) −
detect(arima_var):

| φ (SNR)     | amp 1/(1−φ²) | Δ ×1.5 | Δ ×3 |
|-------------|--------------|--------|------|
| 0.10 (0.05) | 1.01         | 0.00   | 0.34 |
| 0.50 (0.07) | 1.33         | 0.02   | 0.53 |
| 0.70 (0.10) | 1.96         | 0.04   | 0.53 |
| 0.85 (0.18) | 3.60         | 0.10   | 0.20 |
| 0.95 (0.50) | 10.3         | 0.11   | 0.17 |
| 0.99 (2.45) | 50.3         | 0.07   | 0.30 |

**CONFIRMED (subtle ×1.5 break):** (i) Δ → 0 as φ → 0 (Δ = 0.000 at
φ = 0.1: on a white observable, whitening is a no-op and raw ≈ whitened).
(ii) Δ rises with the amplification (Spearman(amp, Δ) = 0.83). (iii)
Secondary consistency check PASSES cleanly: the φ-swept Δ equals the M2
SNR-swept Δ (grid_v5, fixed φ = 0.95) at the *same induced SNR* — 0.11 vs
0.11 at SNR 0.5, 0.07 vs 0.07 at SNR ≈ 2 — so the effect genuinely
operates through the amplified SNR, i.e. the φ-sweep and the SNR-sweep are
one experiment. The state-variance amplification IS the mechanism for the
subtle break.

**FALSIFIED (strong form):** (a) Δ is NOT monotone to φ = 1: for ×1.5 it
peaks at φ = 0.95 (0.11) and DIPS at φ = 0.99 (0.07), because at the
near-unit-root the raw detector's own baseline degrades (calibrated
threshold 1829 at φ = 0.99 vs 275 at φ = 0.95 — the same nonstationarity
penalty as the M4 local-level arena), an opposing effect the simple
1/(1−φ²) law omits. (b) For the coarse ×3 break the prediction fails
outright: Δ is large at every φ including φ = 0.1 (0.34) and
Spearman(amp, Δ) = −0.60 (negative). Raw z² detects a gross q-break
regardless of amplification, while ARIMA whitening attenuates it at all φ.

**Mechanism correction (order check, 80 prefixes/φ).** The ×3 low-φ
ARIMA underperformance is NOT over-differencing: at φ = 0.1/0.5 AIC picks
the *stationary* AR(1) (1,0,0) on 53/51 of 80 prefixes (differencing
(0,1,1) only dominates at φ = 0.99, 58/80). So the ARIMA rung's deficit
is the whitening ITSELF removing the state-shock variance signal that raw
z² retains — the core B2 mechanism — present at all φ, not an
order-selection artifact. This strengthens B2: prewhitening strips the
state-carried signal for q-breaks of any size; the 1/(1−φ²) amplification
only modulates *how visible* the residual raw advantage is on the subtle,
detectability-marginal break, and recedes at the unit root where raw's
baseline breaks down.

**Net.** The user's amplification prediction holds for the regime it
describes (subtle breaks, φ away from 1) and is the confirmed driver
there; it is not a universal monotone law (coarse breaks, and the
near-unit-root boundary, break it). Reported as such in §5. Abstract
carries the robust claim (raw ≥ whitened on q-breaks at every SNR)
without the over-strong tracking clause.

## 2026-07-13 — M4: local-level (RW-state) arena — DEMONSTRATED, not dismissed

`configs/grid_v7_llevel.yaml` (LocalLevelDGP, SNR=q/r ∈ {0.1,0.5,2.0},
level 1σ/3σ + variance ×1.5/×3, T=500, 500 reps); analysis
`experiments/llevel_analyze.py` → `grid_v7_llevel_summary.csv`,
`grid_v7_llevel_degeneracy.png`. The old §3 one-clause dismissal of the
arena is replaced by two demonstrated facts.

**(1) LEVEL detection is degenerate for EVERY method.** At 3σ all five
methods sit at the 5% FAR (raw_cusum 0.07–0.10, Kalman-innov 0.04–0.15,
raw_var 0.06–0.10, ARIMA-var 0.04–0.07, composite 0.07). A level break in
a random-walk state is absorbed by a well-specified filter as one large
ordinary innovation (no sustained signal), and the raw-Y CUSUM has no
fixed baseline — its calibrated threshold is 1533 / 2091 / 2372 (vs
O(10–100) in the AR(1) arena) and it still calibrates hot (FAR 7.8% at
SNR 0.1). No common null exists to rank against; the arena cannot rank
level detectors. The 2026-07-10 dismissal was CORRECT, now with evidence.

**(2) VARIANCE detection is NOT degenerate — but whitening becomes
MANDATORY.** On the ×1.5 variance break the raw z² CUSUM is at chance
(0.06 at every SNR; threshold 10374 / 17463 / 22297 — astronomically
large because z² on a nonstationary RW has no meaningful scale), while
the ARIMA-differencing rung detects (0.71 / 0.84 / 0.58) and the Kalman
composite detects (0.97 / 0.89 / 0.68). At ×3: raw_var stays at chance
(0.06–0.08) while ARIMA-var 0.97–1.00 and composite 0.98–0.99. This is
the exact COMPLEMENT of the AR(1) r-break result (M2): there a raw
variance CUSUM could win when observation noise dominates; here, where Y
is nonstationary, prewhitening is not merely helpful but required — the
raw statistic has no stationary baseline to calibrate against. It
reinforces, rather than contradicts, the paper's central prewhitening
message. FARs 4.8–7.8% (raw_cusum hot at low SNR, as noted).

**Paper action (M6).** §3 dismissal replaced by a short subsection (or
§8.4 expansion) stating (1) and (2) with the degeneracy figure; the
arena now carries a positive finding (whitening-mandatory-under-
nonstationarity), not a hand-wave.

## 2026-07-16 — P1: paper finalization patch (document-only; no grid re-run, no number changed)

**M1 verification pass** — every quoted number checked against its
artifact; all matched: p = 0.007 (rd_eval: 0.0073); the full §5
two-channel ladder (all cells vs ladder_table.csv, exact); the q-break
×3 row 0.72/0.96/0.96 raw and 0.26/0.79/1.00 ARIMA (grid_v5_qbreak);
μ∞ = 0.469 and the Wald delays 68/84/110 (exp06_theory_table). The
`grid_v4_varbench` citation in Appendix C is valid: varbench_ladder.py
regenerates grid_v4_varbench_results.parquet under `make all` by
concatenating the core + T grids. Two text corrections: the
"Albert–Wald" eponym in Prop 2 / Appendix B renamed to the standard
"Wald" first-passage approximation (cites Wald 1947; Siegmund 1985 —
THEORY.md always called it Wald); the training-prefix wording in §2/§4
unified as 125 observations = 25% of T = 500.

**M2 prose (reviewed set)** — abstract (iii) now leads with the ×3
evidence and carries the partial-null hedge (subtle break = "whitening
fails to recover it", not "raw detects it"); §5 q-break reading split
subtle vs coarse with the ≤5-pp-from-FAR chance standard applied to the
×1.5 row and the φ×q Δ values marked as near-floor differences; §6
marks 0.33-at-SNR-2.0 as a single favorable cell; §10 deflationary (1)
softened to "dynamics, at the floor" and the one-sentence answer to
"dynamics (weakly) and attribution (robustly)"; §4 estimation caveat
(ρ̄ = 0.99 is a series-level claim; detection-rate consequences are
empirical); SV/GARCH scope sentence in related work + §10 limitations;
§9 intro states the monitors are distribution-free.

**M3** — Propositions 1–2 now have in-manuscript statements and proofs
in Appendix B (setup/assumptions, mean-path recursion, martingale +
union bound, Wald first-passage derivation); THEORY.md remains the
long-form companion.

**M4** — the three inline-cited figures embedded with captions as
Figures 1–3 (grid_v6_muinf_scatter, grid_v8_phiq_amplification,
grid_v7_llevel_degeneracy); build_paper.py gains implicit_figures and
label-free captions.

**M5** — References section added (17 entries; every inline citation
resolved, none fabricated); abstract compressed to ~220 words
preserving the no/yes/no trichotomy (long version in git history).

## 2026-07-16 — P2: publishability pass (new experiments; science
unfrozen by explicit user request after P1's document-only gate)

P1 froze the science and only patched prose/artifacts. P2 is a genuine
extension: a critical review (prose + gap analysis) identified concrete
weaknesses, and the user authorized new experiments to address them —
this entry logs what was added, run, and found, including where a
first design was wrong and had to be corrected before it reached the
paper.

**Prose/structure (review Bucket A).** Related work split into seven
labeled strands (quickest detection/SPC, innovation-based state-space
monitoring, econometric CUSUM-of-residuals, regime-switching, Great
Moderation empirics, parametric volatility models, offline changepoint)
— was one 45-line paragraph. Contribution 5 (honest-outcome framing)
folded into Contribution 1 (it's a property of the protocol, not a
separate result). Abstract sentence (iii) split from one ~120-word
run-on into three sentences. §7 and §8 Appendix-A internal milestone
tags ("(M2)", "(M7)", "provisional C → B2") stripped or replaced with
self-explanatory text — they were dev-process scaffolding, meaningless
without CHANGELOG context. §8 gained a one-sentence roadmap. §7's
abrupt ending now ties the multi-break lesson back to real recession
clusters (double-dip recessions are exactly the unfixed case).

**B1 retabulation (no new runs, data already on disk).** Added: a
compact ARL₀ table (grid_v1 core) and ARL₁ table to §2 (previously
prose-only, citing `arl_table.csv`/`arl1_table.csv` without showing
them); a numeric Δ-vs-amplification table to §5 (`grid_v8_phiqbreak`,
previously prose-only despite six numbers already being quoted in
running text).

**B2a — PELT as an actual benchmark, not just a related-work dismissal.**
`lsc/benchmarks/changepoint.py::pelt_breakpoints` (ruptures, l2 cost)
already existed but was never run in a grid. New
`experiments/exp08_pelt_benchmark.py`: calibrates PELT's penalty
parameter by bisection to a 5% false-alarm rate on null AR(1) paths
(mirroring the causal-detector calibration protocol exactly), then
evaluates OFFLINE LOCALIZATION (does PELT report a breakpoint within 25
obs of the true break, given the full sample) — explicitly not a delay
comparison, since PELT sees future data the causal detectors cannot.
Result (n=300, `paper_assets/exp08_pelt_results.csv`): PELT is
competitive with the causal raw-Y CUSUM on the canonical 3σ level break
(localize 0.83–0.92 vs. 0.97–0.99) but far weaker on pure variance
breaks (0.00–0.20 vs. the dedicated raw variance-CUSUM's 0.10–1.00),
because PELT's default l2 cost model is fundamentally a mean-shift
statistic. Written up in new §8.5 and the related-work PELT sentence
(previously an unsupported dismissal, now backed by a number).

**B2b — bounded-memory statistics for the §7 multi-break failure.** §7
diagnosed but did not fix the re-arm failure: raw_cusum's fixed-baseline
CUSUM never drains after a permanent shift, so it cannot see a second
event. First design (rejected by its own test before reaching the
paper): restart a Page CUSUM accumulator every `window` observations,
but still compare against the FIXED k/0 reference — this reproduces the
same steady-state elevated value in every window after a permanent
shift and does NOT drain (caught by
`tests/test_multibreak.py::test_windowed_break_pressure_drains_after_
permanent_shift` failing with win[399]=90.7 vs win[150]=77.6, i.e. no
decay at all). Corrected design: a MOSUM-style two-window mean-shift
statistic (Chu, Stinchcombe & White 1996 family) comparing a trailing
window's mean against the window immediately before it — both windows
slide forward together, so `window` obs after a permanent shift both
sit in the same new regime and the statistic returns to its null scale,
while a shift *between* the windows still produces a sharp transient
peak. Implemented as `windowed_break_pressure` (innovation space) and
`windowed_raw_cusum_score` (raw-Y space), wired into `exp04_multibreak.
py` as `lsc_windowed_cusum` / `windowed_raw_cusum` (window=60). Result
(n=500): on the level→level scenario, windowed_raw_cusum's second-event
recall rises from 0.004 to 0.682 — essentially matching its own
first-event recall (0.692) — at higher precision (0.99 vs 0.80) than
the unwindowed statistic, at a modest first-event recall cost (0.692 vs
0.738). The windowed innovation-CUSUM improves less (0.008 → 0.234)
because the filter's own adaptivity already partially forgets a level
shift (μ∞, Proposition 1), leaving less room for a moving reference to
add. On level→variance and variance→variance scenarios BOTH windowed
statistics stay at ≈0.00 — they are mean-shift statistics and a pure
variance change carries no mean signal for a windowed mean comparison
to see. The fix is channel-specific, exactly like §5's main result; a
windowed variance statistic is the natural next step, left to future
work. Existing exp04 methods' numbers are byte-identical to the
pre-P2 committed csv (verified via git diff before committing) — this
is a pure addition, not a re-run of prior results.

**B2c — real-data extensions.** Added a fourth series, `unrate`
(unemployment rate, diff transform, NBER peaks as events, same
treatment as gs10): catches 1973-11, 2007-12, and 2020-02 across most
methods (`paper_assets/rd_unrate_*`). Added a third GS10 event, the
2022-03 hiking-cycle onset (data already pinned through mid-2026, no
re-pull needed for the existing GFC/Volcker events). Added a
false-alarm-rate sweep on INDPRO beyond the existing 5%/10% pair: 1%
(`_far1` tag) and 20% (`_far20` tag), both against the pinned
2026-07-11 snapshot — no network access needed since `load_series`
reads the local snapshot unless `--live` is passed. UNRATE's raw data
snapshot pulled fresh (`data/UNRATE_2026-07-16.csv`, live FRED pull,
network-verified reachable) and pinned for future reruns.
`Makefile:realdata` extended with all of the above;
`experiments/real_data_eval.py` picks up every new run automatically
via its `rd_*_meta.csv` glob.

**B2d — real_data_eval.py permutation-seeding bug, found and fixed.**
Adding the new rd_* runs above shifted rd_indpro's OWN permutation
p-value (lsc_composite: 0.0073 -> 0.0092) even though rd_indpro's
alarm data was verified byte-identical (`git diff` empty on
`rd_indpro_alarms.csv`/`_summary.csv`/`_meta.csv`). Root cause:
`real_data_eval.py` instantiated ONE `np.random.default_rng(20260711)`
and consumed it sequentially across every (run, method) pair in
glob-sorted order — adding a new file earlier in sort order shifts
which random draws land on every later pair, so a run's reported
p-value depended on which OTHER files happened to exist in
`paper_assets/`, not just on its own data. This silently violates the
paper's own reproducibility contract (Appendix A: "make all regenerates
every table... from pinned seeds"). Fixed with a `seeded_rng(run_name,
method)` helper (`zlib.crc32` digest of the pair name folded into a
`SeedSequence`), giving each permutation test a seed that depends only
on its own identity. Re-running with the fix moved every value by
<= 0.002 from the pre-fix estimates (consistent with ordinary Monte
Carlo noise at 20,000 draws, not a systematic shift) — rd_indpro's
lsc_composite lands at p = 0.0080, close to both the pre-P2 committed
0.0073 and the P2-but-still-buggy 0.0092. All in-text citations of
"p = 0.007" updated to p = 0.008 (abstract, §9, §10, Appendix C) with a
one-sentence disclosure in §9 of the fix and its bound on how much any
number moved.

**Reproducibility.** `exp08` added as a new Makefile target and folded
into `make all`; `exp04`'s two new methods run automatically inside the
existing `exp04` target (no Makefile change needed there). Test suite
grew from 95 to 98 (three new tests for the windowed statistics'
bounded-memory property, including the one that caught the first,
wrong design). `make all` reproduces every pre-existing artifact
byte-identically except the already-documented BLAS-epsilon noise in
`m2_param_recovery.csv`.

## 2026-07-23 — four referee-hardening follow-ups: exp15 full grid, paired SE (exp19), pooled exp14 baselines (exp18), composite-on-ARIMA ablation (exp20)

**exp15 GARCH grid extension.** Only ran the subtle ×1.5 break at
SNR ∈ {0.5, 2.0}, on both channels (4 of a 2×2×3 channel × break-size ×
SNR cross). Extended `exp15_garch_benchmark.py` to the full 12-cell
grid, reusing the 4 already-computed cells. **Finding changed, not
just extended**: GARCH is at the false-alarm floor only on the
originally-checked subset (subtle break, moderate/high SNR); it clears
the floor substantially at the coarse ×3 break on both channels
(0.19–0.96) and at r-channel/SNR 0.1 even at ×1.5 (0.498), while
remaining dominated by raw and/or ARIMA in all 12 cells. "GARCH
contributes nothing over chance on this DGP" does not generalize past
the originally-checked cells — Related Work, §10, and Appendix C
updated to the qualified claim.

**exp19 — paired SE for Table 4.** Table 4's Δ = raw − ARIMA advantage
cited a conservative, independence-assuming SE(Δ) ≤ 0.032 bound, but
raw and ARIMA are scored on the same simulated path per replicate
(paired, not independent). `lsc.eval.runner.run` does not persist
per-replicate outcomes (only the aggregated rate), so
`exp19_paired_se_grid_v8.py` reconstructs them by re-running both
detectors (both deterministic given Y — no random restarts in either)
through the identical config/seeds that produced Table 4 — verified to
reproduce all 12 published detect_rate cells exactly, not an
approximation. True paired SEs are 0.014–0.025, 15–55% below the old
bound; the φ=0.95-vs-0.99 subtle-break gap moves from ≈0.8 SE
(indistinguishable from noise) to ≈1.5 SE (suggestive, still not
conventionally significant).

**exp18 — pooled always-raw/always-ARIMA baselines for exp14.** exp14
compared the jointly-calibrated combined statistic against whichever of
raw/ARIMA is best *at each SNR* — an oracle a practitioner facing
unknown SNR cannot use. `exp18_pooled_baseline.py` pools exp14's
existing per-SNR rates under an explicit equal-thirds weighting (no new
simulation): always-raw 0.392, always-ARIMA 0.567, combined 0.480,
oracle-best-per-SNR 0.579 (oracle ≥ both fixed rules at every SNR,
confirmed). Always-ARIMA is the strongest fixed rule and already
captures all but 0.012 of the oracle's advantage — §10's practical
recipe updated.

**exp20 — composite-on-ARIMA ablation.** The ARMA(1,1) equivalence
(exp07) is proven for the innovation series only; 6 of the composite's
11 features are built from the Kalman filtered state, which has no
innovation-series analog. Added `lsc.models.ARIMAModel` (fit-on-prefix
ARIMA wrapped as a `Model`: `fittedvalues` as the state analog,
standardized residuals as the innovations analog) so the EXISTING
`compute_features`/`make_composite_detector` machinery runs unmodified
on ARIMA inputs. Judgment call, disclosed in the script docstring: 5
innovation-based features are a direct, already-precedented
substitution; 6 state-based features are a real interpretive
narrowing (ARIMA has no state distinct from the series it fits). Ran
the same r/q × ×1.5/×3 × SNR{0.1,0.5,2.0} grid Table 3 uses. **Result:
falsifies the naive extrapolation of the innovation-series equivalence
to the full composite.** Away from the detection ceiling the Kalman
composite decisively beats the ARIMA composite (e.g. 0.818 vs. 0.226 at
r ×1.5/SNR 0.1, 11–23 combined SEs); in several cells the ARIMA
composite is even worse than its own single ARIMA-CUSUM feature
(stronger than the previously-documented max-over-features dilution,
§8.3(ii) — there the diluted feature still beat its own null-Adjusted
threshold; here the whole composite underperforms one of its own
inputs). §5 gained a new subsection, the abstract's closing sentence
was qualified, Table 8 added, Appendix C updated. "The ladder is really
raw vs. whitened" now explicitly scoped to the single innovation-series
statistic it was proven for, not extended to the composite.

**Circular-import bug found and fixed while adding `ARIMAModel`.**
`lsc/models/__init__.py` importing `arima_model.py` at module load time
created a cycle: `lsc.diagnostics.features` imports
`lsc.models.base.StateEstimate`, which runs `lsc.models/__init__.py`,
which (with the naive top-level import) imported
`lsc.benchmarks.arima`, which imports back into
`lsc.diagnostics.features` — `ImportError: cannot import name
'break_pressure' from partially initialized module`. Fixed with the
same deferred-import pattern already used by `lsc.eval.detectors` and
`lsc.benchmarks.variance` for this exact `benchmarks.arima` ↔
`diagnostics.features` edge: the `from lsc.benchmarks.arima import
fit_arima_prefix` import moved inside `ARIMAModel.fit()`.

**Runtime anomaly, noted not chased.** `exp20`'s grid took ~8.8 hours
wall-clock, dominated by 3 of 12 cells (q-channel, SNR 0.5/2.0 at
×1.5, and SNR 0.1/0.5 at ×3) each taking 1.5–2.5 hours versus ~2
minutes for every other cell, with `ps` CPU-time accounting showing the
process was NOT continuously CPU-bound during the slow stretches —
consistent with a pathologically slow `statsmodels` ARIMA MLE
convergence on a small subset of the 1,550 per-cell prefix fits
(calibration + scale-estimation + FAR-check + eval), not a hang (every
cell completed with a sane result) and not process contention (no
other heavy process was running concurrently, checked via `ps aux`
mid-run). Unresolved; flagged for anyone rerunning this script that a
per-fit timeout or a faster/bounded optimizer setting may be worth
adding if this recurs.

**Reproducibility.** All four new/modified scripts
(`exp15_garch_benchmark.py`, `exp18_pooled_baseline.py`,
`exp19_paired_se_grid_v8.py`, `exp20_composite_on_arima.py`) and the
new `lsc/models/arima_model.py` are added to the repo; 98 tests still
pass (no new tests added — none of this touched code paths already
covered by the no-lookahead/regression suite, `ARIMAModel` reuses
`fit_arima_prefix`/`ARIMA(...).filter` already exercised by
`lsc.benchmarks.arima`'s own tests). Run strictly one experiment at a
time throughout (lesson from the R1 round's contention bug); no
overlapping heavy sims this round.

## 2026-07-23 (follow-up) — external review of the above: two claims tightened

An external review of the four additions above (full transcript
outside this repo) surfaced two overclaims in the write-up, not in the
underlying computation, and requested one direct check before the
GARCH-grid finding went into the paper as clean.

**exp15: the "empirical FAR = 0.050 in every cell" figure is
tautological.** It is computed on the same calibration draws the
threshold was set from (`(det.null_max_scores >= det.threshold).mean()`),
so it equals the target by construction — true of all 12 cells, old
and new, not a property the grid extension changed or weakened.
`PAPER_DRAFT.md` no longer cites it as if it were an independent check.
The question that actually matters — did the ×3 GARCH reversal come
from scoring detection on the same draws used for calibration — was
checked directly: `run_cell()`'s calibration block draws 500 paths
from a null DGP (no break) at seeds 100000–100499; detection is
evaluated on 500 paths from a break-containing DGP at seeds
200000–200499. Confirmed structurally (disjoint seed sets, different
DGP instances) AND empirically (no calibration path byte-identical to
any evaluation path in a direct sample; substituting a calibration
seed into the evaluation DGP produces a different array than the real
evaluation draw). No data-snooping in the GARCH grid extension.

**exp19: "reproduces the original pairing" overstated what was
verified.** No per-replicate log or column survives from the original
`grid_v8_phiqbreak` run anywhere on disk (confirmed: `grid_v8_
phiqbreak_results.csv` has only aggregate columns, `lsc.eval.runner.
run` discards the per-replicate `outcomes` list after reducing it).
So `exp19_paired_se_grid_v8.py`'s check that the reconstruction
"reproduces the published detect_rate exactly" is a real, verified
claim about the AGGREGATE rate, cell by cell — not a direct check of
the original individual-replicate pairing, which has nothing to check
it against. The pairing claim rests on a determinism argument
(identical seed + code path -> bit-identical Y and bit-identical
scores), itself verified directly (same detector called twice on the
same Y, and Y re-drawn twice from the same seed, both bit-identical)
rather than merely inferred from reading the code. Docstring, inline
comments, and `PAPER_DRAFT.md`'s Table 4 caption all rewritten to
state this precisely: aggregate reproduction is checked; pairing
reproduction is a checked-mechanism argument, not a checked outcome.

## 2026-07-23 -- exp21: innovation-only 5-feature composite isolates where exp20's Kalman-vs-ARIMA gap comes from

exp20 found composite-on-ARIMA badly underperforms composite-on-Kalman
in several cells even though 5 of COMPOSITE_V1's 11 features
(`break_pressure`, `variance_pressure`, `variance_pressure_slow`,
`variance_quiet`, `innovation_ac`) act on `innovations` alone -- an
object exp07's ARMA(1,1) equivalence says the two models should share
on the null path. The other 6 act on the filtered state / one-step
forecast, which has no such shared-object guarantee (ARIMA's
`fittedvalues` standing in for the Kalman filtered state is a
disclosed judgment call, not an equivalence).

New include-list `lsc.diagnostics.features.COMPOSITE_INNOV5` (the 5
innovation-only features) run through the unmodified
`make_composite_detector` on both `KalmanModel("ar1")` and
`ARIMAModel()`, across the identical 12-cell grid as exp20 (channel
{r,q} x vol_mult {1.5,3.0} x SNR {0.1,0.5,2.0}, same seeds/FAR/n_reps
protocol) -- `experiments/exp21_composite_innov5.py`,
`paper_assets/exp21_composite_innov5.csv`.

**Result: the gap opens almost entirely at the innovation-only
level.** Comparing gap = detect(Kalman) - detect(ARIMA) for the
5-feature composite vs. the published 11-feature composite (exp20),
the two agree to within ~0.02-0.03 (the n=500 noise floor) in 10/12
cells, e.g. r/SNR0.5/x1.5: 0.458 vs 0.452; r/SNR2.0/x1.5: 0.280 vs
0.278; q/SNR0.5/x3.0: 0.398 vs 0.380. Only one cell (r/SNR0.1/x1.5)
shows the full composite pulling meaningfully further ahead (0.592 vs
0.442, a real ~0.15 contribution from the six filtered-state
features); two q-channel cells (q/SNR0.1/x1.5 and q/SNR0.1/x3.0) show
the *innovation-only* gap exceeding the full-composite gap -- adding
the filtered-state features narrows the Kalman/ARIMA difference there
rather than widening it. FAR held near the 0.05 target throughout
(0.038-0.066) for both models.

**Reading: destructive substitution, not missing state information.**
Since Y is exactly ARMA(1,1)-equivalent to the Kalman innovations on
the null path (exp07), the fact that ARIMA's own standardized
one-step residual already reproduces almost the whole gap by itself
means ARIMA's residual is a measurably worse detection INPUT under a
break -- not that the composite's power depends on genuinely
state-specific (filtered-state) features the ARIMA substitution lacks.
This narrows exp20's disclosed judgment call: the six filtered-state
features are largely innocent bystanders in the Kalman-vs-ARIMA gap,
not (with one partial exception, r/SNR0.1/x1.5) its cause.

Runtime note: unlike exp20 (~8.8h, dominated by a few pathologically
slow ARIMA MLE fits), this run completed in ~36 minutes -- none of the
12 cells hit that slow-convergence case this time, consistent with
the exp20 CHANGELOG entry's characterization of it as an intermittent
`statsmodels` anomaly rather than a deterministic per-cell cost.

## 2026-07-23 -- exp14/Table 5 reconciliation: calibration-seed bug found and fixed

External review (round 3, "MW2") flagged that exp14's mixed-channel
`arima_var_cusum` detection rates disagreed with the unweighted average
of grid_v5_qbreak's per-channel rates by 1.8 SE at SNR 0.1 growing to
3.8 SE at SNR 2.0, while `raw_var_cusum` agreed within 1 SE throughout
-- asked to find the actual config difference or the bug.

**Found: `experiments/exp14_mixed_channel.py`'s `arima_var_cusum`
calibration used `seed0=200_000`** (four lines below `raw_var_cusum`'s
own calibration call, which correctly used `seed0=100_000`) --
`200_000` is this repo's EVALUATION seed block (`seeds: evaluation:
200000` in every grid config; SPEC's calibration/evaluation/far_check/
feature_scales layout, `experiments/CHANGELOG.md` 2026-07-13), not the
calibration block every other calibration call in the codebase uses.
No comment explained the asymmetry; `raw_var_cusum`'s calibration in
the same script correctly used `100_000`. Compounding it, exp14 also
calibrated on `n_cal=400` reps against the ladder grid's `n_reps=500`.
Both differences apply only to the ARIMA arm's calibration, matching
the reported pattern exactly. Comparing calibrated thresholds directly
(exp14's original CSV vs. `grid_v5_qbreak_far_calibration.csv`):
raw_var_cusum was already close (−2.0%/−5.2%/−7.1% at SNR 0.1/0.5/2.0,
attributable to n=400 vs 500 noise); arima_var_cusum was off by
−19.0%/−9.2%/−13.3% -- a threshold set 9-19% too low, inflating both
the ARIMA arm's false-alarm rate and its detection rate.

**Fix:** `seed0=100_000` for both detectors' calibration, `n_cal=500`
to match the ladder grid exactly (same null DGP/T/n_train per SNR, so
this reproduces grid_v5_qbreak's own thresholds essentially bit-for-bit
where the calibration draws coincide). Verified: rerunning
`exp14_mixed_channel.py` after the fix reproduces
`grid_v5_qbreak_far_calibration.csv`'s thresholds to full precision
(128.693/274.798/615.260 raw and 207.308/173.075/170.232 ARIMA at SNR
0.1/0.5/2.0 -- exact matches, not approximate).

**Result: the discrepancy closes.** Mixed-channel ARIMA detection rate
vs. the unweighted average of the two single-channel ladder rates is
now +0.47 SE (SNR 0.1), +0.53 SE (SNR 0.5), +1.53 SE (SNR 2.0) -- down
from the originally reported 1.8-3.8 SE, and within ordinary Monte
Carlo noise at every SNR. This was a real bug, not a legitimate
convention difference between the two experiments.

**Downstream numbers updated in `PAPER_DRAFT.md`** (§10 practical
recipe, Appendix B summary table): exp14's raw/ARIMA/combined
detection rates changed at all three SNRs (e.g. SNR 2.0: raw
0.213->0.183, ARIMA 0.623->0.560, combined 0.457->0.470); the
"gap widening as SNR rises" framing is dropped since the corrected
gaps are non-monotone (0.050/0.037/0.090 SE-scaled 1.2/0.9/2.2, not a
widening trend) and the SNR-0.5 gap is within 1 SE of zero.
`exp18_pooled_baseline.py` rerun on the corrected `exp14_mixed_channel.csv`:
always-raw 0.374 (was 0.392), always-ARIMA 0.526 (was 0.567), combined
0.490 (was 0.480), oracle 0.549 (was 0.579). The qualitative
conclusions are UNCHANGED (always-ARIMA still beats "run both" and
"always raw" pooled; ARIMA still wins 2 of 3 SNRs; oracle's edge over
always-ARIMA still small) but the margins shrank (always-ARIMA's edge
over the combined statistic: 0.09 -> 0.036, now only ~1.5 SE) --
"running both is only clearly justified if..." framing is unaffected,
but the strength of the "always-ARIMA is a stronger fixed rule" claim
is now stated as modest rather than the 0.09 gap it previously read.

Reproducibility: `experiments/exp14_mixed_channel.py` fix and rerun
(269s, n_eval=300); `experiments/exp18_pooled_baseline.py` rerun
(reads the updated CSV, no simulation). Both outputs committed under
`paper_assets/`.

## 2026-07-23 -- exp22: threshold/argmax diagnostic resolves MW3 (exp21 full table + noisy-substitute question)

External review round 3 ("MW3") asked for (a) the full 12-cell
exp21_composite_innov5.csv published in the paper, not just the three
cells previously described only in this CHANGELOG, and (b) a direct
check, at r x1.5/SNR 0.1 specifically, of whether the ARIMA composite's
calibrated threshold is substantially higher than the Kalman
composite's -- which would point to a "noisy substitute" reading (the
six ARIMA-fed filtered-state-analog features inflate the null score
distribution) rather than exp21's "destructive substitution... traces
to the innovation series" framing.

**Table 9 (full exp21 grid) added to PAPER_DRAFT.md**, same treatment
as Table 8.

**New script `experiments/exp22_composite_threshold_argmax.py`**
reconstructs both composites with the exact recipe behind Table 8 at
r x1.5/SNR 0.1 (500 calibration reps, 500 break-path evaluations) and
reports (i) the calibrated threshold for each and (ii) the
argmax-feature distribution at alarm time for each, using the same
`composite_attribution` helper `real_data.py` already uses for
real-data alarm attribution.

**Result: both readings hold, at different levels of the mechanism.**
Threshold: ARIMA composite 45.49 vs. Kalman composite 35.28, +28.9% --
a real, substantial gap, evidence for the noisy-substitute reading (the
same shared-threshold dilution mechanism already documented in
Sec 8.3(ii) for a different composite variant). Argmax-at-alarm
distribution: Kalman composite's 415 alarms (n=500) are 96%
`variance_pressure` (an innovation-only feature); ARIMA composite's 124
alarms are 77% `variance_pressure` + 15% `break_pressure` (both
innovation-only) -- the 6 disputed filtered-state-analog features
account for only 4-7% of alarms in either composite. So the six extra
features are rarely what actually FIRES on a true break (supporting
exp21's innovation-only framing) but do measurably inflate the shared
null max-score distribution the composite's threshold is calibrated
against (supporting the noisy-substitute framing) -- both mechanisms
are real and distinct, operating at different stages (which feature
detects vs. which features set the bar). PAPER_DRAFT.md's §5 discussion
updated to state this precisely rather than picking one reading.

Reproducibility: exp22 run once (500 reps, ~200s, no ARIMA-order-search
pathology hit); outputs committed under paper_assets/
(exp22_composite_threshold_argmax.csv, exp22_summary.csv,
exp22_thresholds.csv).

## 2026-07-23 -- exp24: independent GARCH FAR check resolves MW4

External review round 3 ("MW4") noted exp15's GARCH grid was the one
arm not yet held to the paper's own "empirical FAR re-verified on
fresh nulls" standard: the disjoint calibration/evaluation seed check
(2026-07-23 entry above) rules out data-snooping between calibration
and evaluation but doesn't establish the calibrated threshold delivers
5% FAR out of sample, which matters more for GARCH given its heavier-
tailed order-statistic threshold (Sec 8.4's general caveat).

**New script `experiments/exp24_garch_fresh_far_check.py`** reproduces
each of exp15's cells' exact calibration (seed0=100000, n_reps=500,
same DGP/T/n_train) then evaluates the resulting threshold on 500
fresh null draws from the project's standing far_check seed block
(300000+, disjoint from both calibration and evaluation by
construction, per the SPEC's seed layout). Since calibration and the
null-only FAR check depend only on SNR (not channel/vol_mult -- the
null DGP has no break), computed 3 times (one per SNR) and replicated
across the 4 channel/vol_mult combinations to match exp15's 12-row
grid shape.

**Result: no anomaly.** GARCH fresh-draw FAR = 5.4% / 5.0% / 4.8% at
SNR 0.1/0.5/2.0 (binomial SE ~1.0pp at n=500) -- within 0.6pp of the 5%
target at every SNR. raw_var_cusum and arima_var_cusum fresh-draw FARs
also checked in passing (raw: 6.0/6.6/6.0%; arima: 4.4/6.0/4.2%), both
similarly close to target. PAPER_DRAFT.md's GARCH benchmark discussion
(Sec 3/Related Work) updated with this result, distinguishing it
explicitly from the pre-existing tautological same-draw FAR figure.

Reproducibility: exp24 run once (500 reps x 3 SNRs, ~196s total, no
slow-ARIMA-convergence pathology hit); output committed under
paper_assets/exp24_garch_fresh_far_check.csv.

## 2026-07-23 -- exp23 + scope decision resolves MW5 (real-data look-ahead boundary, vintage coverage)

External review round 3 ("MW5") asked for two things: (a) print the
real-data pipeline's actual train/bootstrap/monitor index boundaries
and confirm the bootstrap DGP is fit on the training prefix only, plus
run the existing bit-identical perturbation test against the real
pipeline directly; (b) extend the ALFRED real-time vintage protocol
beyond INDPRO's GFC/COVID alarms, or explicitly label the other tables
as revised-data-only.

**(a) New script `experiments/exp23_realdata_lookahead_check.py`.**
Confirmed by direct code read (`real_data.py:162`, `null =
fitted_null(Y[:NT])`) that the bootstrap null's AR(1) parameters come
from the training prefix only, and by computation that this actually
matters (INDPRO's GFC segment: phi=0.954/q=0.0046 fit train-only vs.
phi=0.892/q=0.0385 on the full segment -- these are not close). Then
ran the tests/test_no_lookahead.py-style bit-identical perturbation
check against the REAL pipeline's per-segment procedure for the first
time (corrupt the monitored window past a point t, rerun bootstrap-fit
+ calibrate + score, compare to the uncorrupted run): threshold AND
score-prefix bit-identical for all five real-data detectors on INDPRO
segment 10 (the GFC segment). The earlier simulated-DGP no-lookahead
test never exercised the real pipeline's threshold-SETTING step
specifically; this does, and it passes. PAPER_DRAFT.md's Sec 9 intro
now states this directly rather than only asserting it from the code's
structure.

**(b) Vintage coverage: labeled, not extended.** Confirmed by live
query (2026-07-23) that ALFRED serves vintage histories for GDPC1,
GS10, and UNRATE (not just INDPRO), so the extension is technically
available. Did not attempt it this round: a full per-series
episode/decision-month grid with its own recalibration at three
different training-window lengths is a materially larger undertaking
than the checks above, and this project's own history (window-
anchoring bug in exp13c, GDP quarter/month units mismatch in exp13d,
both in Supplementary Materials) shows rolling-window protocol
extensions done under time pressure have twice introduced real bugs
only caught by a later dedicated check. Chose the paper's other
option instead: Table 6 and the real-time discussion in Sec 9 now
explicitly state that only INDPRO's GFC/COVID alarms are vintage-
verified and everything else (all of GDP/GS10/UNRATE, and Table 7's
sensitivity variants) is a revised-data illustration. Flagged as a
well-scoped but not-yet-executed follow-up, not silently deferred.

Reproducibility: exp23 run once (light -- Kalman-only detectors, no
ARIMA order search, <10s); output committed at
paper_assets/exp23_realdata_lookahead_check.txt.

## 2026-07-23 -- exp25: ICSS benchmark added, resolving the "missing experiment" item

Peer review round 3 (Missing Experiments) flagged ICSS (Inclan & Tiao
1994) as the conspicuously missing variance-changepoint counterpart to
the existing PELT (mean-shift) benchmark in Sec 8.5 -- the paper
already documents PELT's mean-shift cost model performing poorly on
variance breaks (0.00-0.20 localization), which invites the "wrong
tool" objection without a purpose-built alternative to check.

**New `lsc.benchmarks.changepoint.icss_breakpoints`**: the standard
ICSS recursive search (D_k = C_k/C_T - k/T, C_k = cumsum(seg**2)[k],
partition at argmax|D_k| when it exceeds a threshold, recurse on both
halves). Threshold (`crit`) calibrated by simulation to a target FAR
via bisection, the same calibrated-parity convention as PELT's `pen`
-- not Inclan-Tiao's asymptotic critical value, since the threshold is
set empirically anyway and the raw D_k statistic is already scale-free.

**New `experiments/exp25_icss_benchmark.py`**, mirroring exp08's
design exactly: offline localization (full standardized post-training
segment, not a causal delay comparison, same exclusion as PELT per
SPEC Sec 4.1), same arenas/seeds/window (+-25 obs) as exp08, restricted
to the variance scenarios (both channels: r/"variance" and
q/"state_var") since ICSS has no mean-shift claim. n=500 (vs. exp08's
300 -- ICSS has no per-replicate model fit, cheap enough that matching
the rest of the paper's standard n_reps cost nothing; full 12-cell run
took 5s).

**Result: ICSS clears PELT's ceiling but is still dominated by the
causal detector.** ICSS localizes up to 1.00 on variance breaks
(vs. PELT's 0.00-0.20 on the same scenarios) -- confirming the earlier
PELT gap was specifically its mean-shift cost model, not offline
methods generally. But despite an unfair advantage (full 375-obs
post-training segment visible at once, no causal constraint), ICSS is
dominated by the causal raw_var_cusum in 11 of 12 cells, tying only at
r x3/SNR 0.1 (both at ceiling) -- e.g. r x1.5: ICSS 0.74/0.06/0.00 vs.
raw_var_cusum 0.996/0.560/0.102 over SNR 0.1/0.5/2.0. The SNR-dependent
collapse pattern matches Outcome C's already-documented mechanism
(state-driven autocorrelation swamping a shrinking noise-variance
signal as SNR rises), and ICSS is hit by it even harder than the
causal CUSUM. Table 5b added to PAPER_DRAFT.md (Sec 8.5, immediately
after Table 5/PELT) with full discussion; Appendix C updated.

Reproducibility: exp25 run once (500 reps/cell, 5s total, no
per-replicate model fit); outputs committed under
paper_assets/exp25_icss_results.csv,
paper_assets/exp25_icss_far_calibration.csv.

## 2026-07-23 -- exp26 + exp27: known-parameter variance ladder and windowed variance statistic (final two Missing Experiments)

Peer review round 3 (Missing Experiments) asked for two more additions
beyond ICSS: a known-parameter column throughout the variance ladder
(exp10 only ever checked one level-shift cell), and a windowed
variance statistic to close the one gap the existing MOSUM-style
mean-shift fix leaves open (Sec 7's var_up_down second-event miss).

**exp26 (known-parameter variance ladder).** New
`lsc.benchmarks.variance.known_raw_var_cusum_score` (standardize by
the DGP's analytic stationary SD, sqrt(q/(1-phi^2)+r), instead of the
training-prefix sample SD) and `known_kalman_var_cusum_score` (the
same three-arm CUSUM on `lsc.theory.steady_state_innovations` instead
of an MLE-fit KalmanModel's innovations). Run across the identical
12-cell grid Table 3/5 uses (`experiments/exp26_known_param_variance.py`,
n=500, ~50s total, no ARIMA fits). Result: 10/12 raw-rung and 9/12
Kalman/ARIMA-rung cells show known >= estimated (negative cells all
within MC noise, near ceiling). Two findings worth flagging: (1) r
x1.5/SNR0.5 -- squarely on Outcome C's SNR-collapse curve -- nearly
closes under known parameters (0.560->0.964, the largest gap in the
table), while SNR 2.0 still collapses even known (0.102->0.168) --
so Outcome C's autocorrelation-masking mechanism explains the SNR-2.0
floor but not the full steepness of the SNR-0.5 midpoint, which is
substantially an estimation artifact; (2) q-channel Kalman/ARIMA gaps
are large and one-sided (+0.05 to +0.56) -- AIC-order-selection/MLE
noise contributes to arima_var_cusum's underperformance on top of the
model-class gap exp07's ARMA(1,1) equivalence already predicts is zero
on the null path. Table 2b added to PAPER_DRAFT.md Sec 5.

**exp27 (windowed variance statistic).** New
`lsc.benchmarks.variance.windowed_raw_var_score` /
`lsc.eval.detectors.make_windowed_raw_var_cusum_detector`: a
two-window log-variance-ratio statistic (log(test_var/ref_var)
rescaled by its delta-method SE, sqrt(4/window)), the variance-channel
mirror of the existing mean-shift windowed_raw_cusum_score. Tested on
the EXACT var_up_down scenario exp04 already uses (obs-noise x3 at
t=200, x1/3 at t=350, 150-obs spacing, same arena/seeds/re-arm
protocol; `experiments/exp27_windowed_variance.py`, n=500, 12s).
Result: closes the gap. raw_var_cusum (fixed-baseline): recall_break1
0.998 / recall_break2 0.000 (never drains, as already documented).
windowed_raw_cusum (existing mean-shift fix): 0.000/0.000 (no mean
signal at either break for a mean-comparison statistic to see).
windowed_raw_var (new): recall_break1=0.932, recall_break2=0.948,
F1=0.958, precision=0.997 -- both events well detected, no
first/second asymmetry. Sec 7 rewritten: the "closing that gap needs a
windowed variance statistic... left to future work" sentence is now a
reported result, not an open problem; the double-dip failure mode is
reframed as "wrong bounded-memory statistic for the channel," not a
structural limit.

Reproducibility: both scripts run once at their stated n_reps; outputs
committed under paper_assets/exp26_known_param_variance.csv,
paper_assets/exp27_windowed_variance.csv,
paper_assets/exp27_windowed_variance_far.csv.

## 2026-07-23 -- fixed a real gap: four verified citations never added, one used but missing from References

External review round 3 supplied six verified citations. Four (Aue &
Horvath 2013, Aue & Kirch 2024, Andreou & Ghysels 2002, Berkes/Gombay/
Horvath/Kokoszka 2004) were never added to the paper at all -- no task
in the round-3 punch list covered them explicitly, and they were
missed. A fifth, Inclan & Tiao (1994), was used in-text (added this
same round, for the ICSS benchmark, Sec 8.5) but never added to the
References list -- a citation with no bibliography entry. Caught by
the external reviewer re-checking the pushed commit, not by this
repo's own process.

Fixed all six, each with a real in-text citation (not just a
reference-list entry):
- Aue & Kirch (2024) and Aue & Horvath (2013): added to the "Quickest
  detection and SPC" Related Work paragraph (Sec 1) -- the former as
  the direct survey of the CUSUM family this paper uses throughout,
  the latter as the broader structural-break literature survey.
- Berkes, Gombay, Horvath & Kokoszka (2004) and Andreou & Ghysels
  (2002): added to the GARCH Related Work paragraph (Sec 1), as
  existing break-aware/multiple-break GARCH literature the paper's
  "remains open" sentence had no citation for.
- Harvey & Koopman (1992): added to Sec 9's UNRATE model-fit-check
  paragraph, as the classical diagnostic-checking-of-unobserved-
  components reference justifying why per-window (not just
  on-average) parameter/residual checks are the right diagnostic.
- Inclan & Tiao (1994): added to the References list (in-text
  citation already present from the ICSS addition).

All six now appear both in-text and in the alphabetized References
list, in the paper's existing citation format.

## 2026-07-24 — R2 M1 PRE-REGISTERED: φ = 0.99 as a second operating point (r-channel + known-parameter), before any new grid cell runs

Registered before implementing `configs/grid_v9_r_phi99.yaml` or the
φ = 0.99 known-parameter script (verified: neither file exists at
commit time). The paper's headline body arena uses φ = 0.95
throughout; φ = 0.99 is a substantially more persistent, arguably more
empirically realistic operating point (§4's `grid_v6_phisweep` already
sweeps φ ∈ {0.5, 0.8, 0.95, 0.99} for the level-shift innovation-CUSUM,
and `grid_v8_phiqbreak` already has a φ = 0.99 row for the q-channel
raw-vs-ARIMA ladder at fixed q = 0.04875, r = 1.0). Two cells have
never been run at φ = 0.99: (i) the r-channel (observation-noise)
whitening ladder — raw_var_cusum vs arima_var_cusum, ×1.5 and ×3
obs-noise breaks, SNR ∈ {0.1, 0.5, 2.0} via q = SNR·(1−φ²)·r (the
grid_v4 convention; at φ = 0.99 this gives q ∈ {0.00199, 0.00995,
0.0398}, matching the q values already used in `grid_v6_phisweep`'s
φ = 0.99 row) — and (ii) the known-parameter counterpart
(`known_raw_var_cusum_score` / `known_kalman_var_cusum_score`, exp26's
method) for BOTH channels at φ = 0.99, extending Table 3's
known-vs-estimated ablation the same way exp26 did at φ = 0.95. The
q-channel estimated ladder at φ = 0.99 is NOT rerun — `grid_v8_phiqbreak`
already covers it (arena `ar1_phi0.99`, q = 0.04875 fixed, induced
SNR ≈ 2.45) and is pulled into the comparison table directly.

**Pre-registered prediction (falsifiable).** The paper's trichotomy —
(a) r-breaks: raw is SNR-dependent and whitening rescues it; (b)
q-breaks: raw matches or beats whitening, an inverted ordering; (c)
known parameters narrow but do not close the estimated-rung gaps —
holds QUALITATIVELY at φ = 0.99: raw_var_cusum's r-channel detection
rate still falls with SNR, arima_var_cusum stays flatter, and the
q-channel ordering (raw ≥ ARIMA) still holds at ×3. **Falsifiers:**
the r-channel SNR-dependence reverses sign or vanishes at φ = 0.99; the
q-channel ordering inverts back (ARIMA beats raw) at ×3; known
parameters close an estimated-rung gap that was open at φ = 0.95 (or
vice versa) at φ = 0.99. A φ-dependent qualifier is explicitly an
acceptable, publishable outcome per Proposition 1 (μ∞ and the
fast-or-never boundary are both φ-dependent) — not a failure to be
smoothed over. Same protocol as grid_v4/grid_v8: n_reps = 500,
T = 500, far_target = 0.05, train_frac = 0.25, identical seed blocks.
Outcome logged with numbers when resolved.

## 2026-07-24 — R2 M1 RESOLVED: CONFIRMED IN PART, FALSIFIED IN PART (honest mixed, same pattern as M7)

`configs/grid_v9_r_phi99.yaml` (r-channel, φ=0.99, both break sizes,
three SNRs) + `experiments/exp28_known_param_phi99.py` (known-parameter
counterpart, both channels) + `experiments/phi99_robustness_table.py`
(assembler). Full numbers: `paper_assets/grid_v9_r_phi99_results.csv`,
`paper_assets/exp28_known_param_phi99.csv`,
`paper_assets/phi99_robustness_{estimated,known}.csv`. Table 3b added
to PAPER_DRAFT.md (§5, after Figure 2/Table 4).

**Falsified**: the subtle ×1.5 r-break's estimated-rung ordering does
NOT survive φ=0.99 — ARIMA is not flatter than raw as predicted, it's
lower and non-monotone (0.58/0.20/0.27 vs raw 0.98/0.24/0.07 across
SNR 0.1/0.5/2.0), losing to raw at 2 of 3 SNRs. **Confirmed**: the
coarse ×3 r-break ordering (ARIMA >= raw) and the q-channel ordering
(raw >= ARIMA) both survive, the latter more decisively than its
closest φ=0.95 comparator (q x3: raw 0.58 vs ARIMA 0.28 at φ=0.99,
vs. raw 0.96 vs ARIMA 1.00 — ARIMA winning — at φ=0.95/SNR2.0).

The known-parameter ablation (exp28, same method as exp26) diagnoses
the subtle r-break falsifier as estimation, not mechanism: known-Kalman
stays flat at 0.984 across SNR at BOTH φ (matching φ=0.95's
0.986/0.984/0.984) -- whitening's POPULATION-level case for the r
channel is intact, if anything strengthened, since known-raw itself
now falls sharply with SNR at φ=0.99 (0.990/0.390/0.062) where it was
roughly flat at φ=0.95 (0.988/0.964/0.168). The known/estimated ARIMA
gap at φ=0.99 (+0.40 to +0.78) dwarfs the largest φ=0.95 gap (+0.40,
exp26) -- near-unit-root AIC order-selection / MLE difficulty (already
flagged as a "benign" artifact at φ=0.95 in the exp07 ARMA-equivalence
discussion) is not benign at φ=0.99. Same diagnosis on the coarse
r-break (raw's estimated decline at SNR2.0, 0.56 vs known-raw's
near-ceiling 0.97, is also mostly estimation) and the q channel
(known-Kalman beats known-raw at BOTH φ -- the population-level q-story
was never "raw wins"; the φ=0.99 estimated advantage for raw is driven
even more by ARIMA's near-unit-root fragility than the ARMA-θ-shift
mechanism Table 3 credits at φ=0.95).

Headline: the r-channel "prewhitening wins" claim in Table 3 is an
ESTIMATED-rung statement, not a population-level one, and specifically
fragile near the unit root because ARIMA order-selection/estimation
degrades there -- not because whitening stops helping. Publishable
scope qualifier, not a result to smooth over; matches the falsifier
clause pre-registered above almost exactly (the r-channel SNR-dependence
did not reverse sign, but the whitening advantage it was paired with
did, at the subtle break specifically).

## 2026-07-24 — R2 M2 PRE-REGISTERED: AR(2)+noise as a second DGP, before any implementation

Registered before writing `AR2StateDGP` (verified: no such class exists
in `lsc/dgp/` at commit time). The paper's entire theoretical apparatus
(Propositions 1–2, the ARMA(1,1) equivalence, exp07) is derived for
AR(1)+noise specifically. AR(2)+noise — S_t = φ₁S_{t-1} + φ₂S_{t-2} + w_t,
Y_t = S_t + v_t, same additive observation noise — is a minimal but
genuine generalization test: a second-order persistence structure
reducible in principle to a closed-form ARMA representation, but not
the same one-to-one identity Prop. 1/exp07 prove for AR(1), so it tests
whether the empirical trichotomy (level: raw wins; r-breaks: whitening
wins; q-breaks: raw wins) survives outside the exact algebraic
correspondence the paper leans on.

Two parameterizations, chosen for real vs. complex characteristic
roots (qualitatively different persistence regimes, run and reported
separately, not pooled): (i) real, well-separated roots — e.g.
φ₁ = 1.4, φ₂ = −0.45 (roots ≈ 0.5, 0.9, both real, stationary since
both |root| < 1); (ii) complex roots — e.g. φ₁ = 1.6, φ₂ = −0.9 (complex
roots with modulus ≈ 0.949, giving oscillatory/quasi-cyclical
persistence). Both stationarity-checked (roots of 1 − φ₁z − φ₂z² = 0
outside the unit circle) before use. **Disclosed modeling choice, not
a default:** the q-channel ("state-innovation variance") break for
AR(2) scales the SD of the single shock w_t by `vol_mult` — unambiguous,
since there is one shock — but AR(2) has TWO autoregressive
coefficients, so "which channel counts as the state-innovation one"
is a real choice, not a natural extension of the AR(1) definition; we
use the innovation-variance definition (scales w_t's SD) because it is
the direct structural analogue of AR1StateDGP's q-break and requires
no further choice about which φ to perturb (a `persistence`-type break
on one of φ₁/φ₂ is a DIFFERENT, unimplemented question, left open).

One representative SNR and break size per parameterization (not the
full grid): SNR 0.5, ×1.5 (r/q channels) or 1σ (level), matching the
paper's most-discussed subtle-break cells. Core trichotomy comparisons
only: level-shift raw CUSUM vs. innovation CUSUM; r-break raw_var_cusum
vs. arima_var_cusum; q-break raw_var_cusum vs. arima_var_cusum.

**Pre-registered prediction (falsifiable).** The qualitative ordering
from the AR(1) trichotomy holds under both AR(2) parameterizations:
raw beats/matches innovation-CUSUM on the level break; arima_var_cusum
beats raw_var_cusum on the r-break; raw_var_cusum matches or beats
arima_var_cusum on the q-break. **Falsifiers:** any of the three
orderings flips sign under either parameterization. A flip is an
honest scope-limit finding (the AR(1)-specific machinery not
generalizing), reported the same way as a confirmation, not smoothed
over. Same protocol conventions as the rest of the paper: n_reps = 500,
T = 500, far_target = 0.05, train_frac = 0.25, seed blocks disjoint
from all published grids (calibration 110000+, evaluation 210000+, to
avoid seed collision with any existing arena). Outcome logged with
numbers when resolved.

## 2026-07-24 — R2 M2 RESOLVED: fully CONFIRMED, no falsifiers

New `AR2StateDGP` (`lsc/dgp/continuous.py`; registered in
`lsc/dgp/__init__.py` and `lsc.eval.runner.DGP_CLASSES`), new `'ar2'`
`KalmanModel` spec (`lsc/models/kalman.py`, `UnobservedComponents(...,
autoregressive=2)`), tests `tests/test_ar2.py` + AR2 entries added to
`tests/test_dgp.py::ALL_DGPS` (stationarity of both parameterizations,
sigma_ref vs. empirical variance on a 20k-step path, level/variance/
state_var break conventions carried over unchanged from AR1StateDGP).
`experiments/exp29_ar2_trichotomy.py` runs the core comparison; full
numbers `paper_assets/exp29_ar2_trichotomy.csv`. Table 6 added to
PAPER_DRAFT.md §8.6.

All six pre-registered cells confirm the AR(1)-derived ordering: level
(raw_cusum beats innovation_cusum, real roots 0.376 vs 0.156, complex
roots 0.928 vs 0.760), r-channel (arima_var_cusum beats raw_var_cusum,
real roots 0.968 vs 0.660, complex roots 0.938 vs 0.810), q-channel
(raw_var_cusum beats arima_var_cusum, real roots 0.276 vs 0.184,
complex roots 0.420 vs 0.280). No falsifiers triggered. The only one of
this round's three extensions (R2 M1's φ=0.99, this, Appendix A's
cross-environment check) to come back clean with no qualification
needed. Scope caveat (not a falsifier): one cell per parameterization
at one SNR/break-size is an existence check that the ordering survives
leaving AR(1), not a characterization of how it varies across the
AR(2) parameter space.

## 2026-07-24 — R2 M3: cross-environment reproduction (design registered and resolved together — infrastructure check, not a falsifiable hypothesis, same category as the m6x real-data extension's design registration)

Registered and run in the same session: build a container (`Dockerfile.repro`,
`.dockerignore`, colima + Docker on the author's machine) deliberately
different from the development environment on OS (Linux/Debian vs.
macOS), C library (glibc vs. libc/Accelerate), and Python minor version
(3.12 vs. 3.14 host), installing dependencies ONLY from
`pyproject.toml`, then run `make all` with zero manual intervention and
diff every output against the committed `paper_assets/`. Explicitly
scoped as CROSS-ENVIRONMENT, not third-party: still the author's own
tooling, just a different container — labeled that way throughout
rather than left to imply the stronger claim. Full writeup: PAPER_DRAFT.md
Appendix A, "A cross-environment reproduction" (new subsection).

**Three real bugs found and fixed before a clean run, each a genuine
gap a same-machine check cannot surface:**
1. `pyproject.toml` declared `requires-python = ">=3.11"` but pinned
   `numpy==2.5.1` needs Python >=3.12 -- first build failed outright.
   Fixed: `requires-python` corrected to `>=3.12`.
2. `experiments/m2_param_recovery.py`'s `to_latex()` needs `jinja2`
   (pandas 3.x routes it through `Styler`) -- undeclared; only "worked"
   on the host because jinja2 was present from an unrelated, unrecorded
   install (`pip show jinja2` on host: `Required-by:` empty). Fixed:
   added `jinja2==3.1.6` to dependencies.
3. First `Dockerfile.repro` piped `make all | tee`, whose exit code
   (always 0) masked make's real failure -- container reported success
   while make had actually failed (paper_assets/ excluded from the
   build context via `.dockerignore`, so there was nowhere to write).
   Fixed: paper_assets/ now copied into the image (matching a real
   `git clone`, which make then overwrites in place); run redirects to
   a log file instead of piping through tee, so failures now surface
   as the container's real exit code.

**Result after fixes: clean run (`make all` exit 0), diffed file-by-file
against committed paper_assets/** (excluding m2_param_recovery.csv,
already-documented BLAS-thread-order nondeterminism, lesson 11). 18/23
`*_results.csv` files byte-identical; the other 5 show detect_rate
differences of at most 0.006 (3/500 reps), confined to MLE-fit-dependent
methods (lsc_composite, lsc_state_cusum, ARIMA rungs) -- raw_cusum and
other closed-form statistics are exact everywhere. Mechanism: almost
certainly a different BLAS/LAPACK backend (macOS Accelerate vs.
container OpenBLAS) under identical pinned numpy/statsmodels versions,
nudging iterative-optimizer convergence by ~1 ULP-class amounts -- the
same nondeterminism class as lesson 11, evidently larger (but still
small relative to any reported effect size) across a genuine library
change rather than repeated runs on one machine. exp29_ar2_trichotomy.csv
(R2 M2) and grid_v9_r_phi99's results (R2 M1) -- the two genuinely
fresh, non-cached computations this round added -- are BYTE-IDENTICAL
end-to-end: the strongest single piece of evidence, a full
simulate->fit->detect->calibrate pipeline reproduced to the last bit
across OS/libc/Python version. exp28_known_param_phi99.csv shows only
last-digit CSV string-formatting differences on rows its own
`_already_done` cache reused unchanged -- a serialization artifact, not
a value difference.

Confound noted for transparency, not correctness: mid-run, host CPU
load spiked to 50 (other processes on the author's own machine,
unrelated to this container) and the run slowed ~20-30x for about 90
minutes before easing -- affects wall-clock only.

Honest scope of the claim: closed-form statistics reproduce to the
literal bit across environments; MLE-dependent ones reproduce the
substantive finding (detect_rate within 0.006, an order of magnitude
below any effect size this paper calls meaningful) but not literal
bit-identity -- floating-point optimizer convergence is not portable
across BLAS backends even with every package version pinned. Real,
disclosed limit, not smoothed over. Separately: this is still NOT a
third-party reproduction (independent person, cold clone, README only)
-- none has been performed; flagged as the more credible addition if a
labmate/advisor becomes available before submission, per the
scoping decision made when this round began.

## 2026-07-25 — R3 PRE-REGISTERED: order-known ARIMA rung (M1) + formal paired test for the phi-sweep/SNR-sweep equivalence (M2), before either script is run

Two reviewer-requested gaps, registered together before implementation
(verified: no exp30/exp31 files exist at commit time).

**Correction to the requested seed convention (M1), flagged before
running rather than silently substituted.** The request specified a
fresh disjoint seed block (300000+) for exp30's order_known condition.
That block is already reserved, project-wide, for a DIFFERENT purpose:
`experiments/CHANGELOG.md` (2026-07-13) fixes the standing layout
calibration=100000, evaluation=200000, far_check=300000,
feature_scales=900000 -- reused IDENTICALLY across every grid
specifically so cells are draw-for-draw comparable (stated explicitly
in grid_v4/v5/v8's config headers; exp19 reconstructs per-replicate
pairing by rerunning with these SAME seed bases, verified against
published aggregates). exp30's own stated sanity check --
`gap_order_selection + gap_coefficient_noise` should equal the
published `detect(known) - detect(estimated)` -- REQUIRES order_known
to be evaluated on the identical simulated paths as `estimated`
(grid_v5's arima_var_cusum) and `known` (exp26's known_kalman rung):
using a fresh, unrelated seed block would make the three conditions
independent Monte Carlo estimates rather than a decomposition, so the
sanity check could fail (or pass by chance) for reasons having nothing
to do with order-selection vs. coefficient noise. exp30 therefore
reuses the standing calibration=100000/evaluation=200000 blocks (same
as exp26/grid_v5), and uses far_check=300000 ONLY for the requested
fresh-null FAR re-verification, exactly matching exp24's convention
(itself built on the identical standing layout) -- not a new block.

**M1 design.** Three-way decomposition on all 6 published q-channel
Table 3 cells (channel=q, vol_mult in {1.5, 3.0}, SNR in {0.1, 0.5,
2.0}, phi=0.95, T=500, n_train=125, n_reps=500, 5% calibrated FAR):
`estimated` (existing, AIC order + MLE coefficients, grid_v5's
arima_var_cusum), `order_known` (NEW: order fixed at true (1,0,1), MLE
coefficients -- `order_known_var_cusum_score`,
`lsc/benchmarks/arima.py::fit_arima_prefix_fixed_order`), `known`
(existing, exp26's known_kalman_var_cusum_score). Falsifiable check:
gap_order_selection + gap_coefficient_noise should equal the published
known-minus-estimated gap to within rounding; a real mismatch would
indicate a seed/protocol divergence between exp30 and exp26, not a
substantive finding, and will be reported as such rather than folded
into the headline numbers.

**M2 design.** Formal test for "the phi-sweep and SNR-sweep are one
experiment" (currently a two-decimal eyeball match) at the two matched
induced-SNR points already cited in the text. Point 1 (SNR=0.5): NOT
run fresh -- grid_v5's `ar1_snr0.5` arena (phi=0.95, q=0.04875, r=1.0)
and grid_v8's `ar1_phi0.95` arena (phi=0.95, q=0.04875, r=1.0) are
VERIFIED IDENTICAL by construction (same DGP parameters, same
calibration/evaluation seed bases; grid_v8's own config header states
the phi=0.95 anchor was chosen to reproduce the SNR=0.5 body arena) --
checked directly against the committed CSVs: detect_rate and
mean_delay_censored are bit-identical for both q-channel scenarios at
this point. This is not "two experiments that happen to agree," it is
the same computation appearing in both grids; no hypothesis test
applies, and the writeup will say so plainly rather than present it as
independent confirming evidence. Point 2 (SNR=2.0 vs. grid_v8's
induced SNR=2.45 at phi=0.99): genuinely different DGP parameterizations
(different phi AND q) -- the requested shared-seed pairing does not
apply here (identical seed integers feed different (phi, q) transition
dynamics, so they do not produce exchangeable pairs), so this point
uses the permutation-test fallback: reconstruct per-replicate detection
outcomes for both cells by rerunning raw_var_cusum/arima_var_cusum
through the ORIGINAL seed bases (exp19's methodology, verified against
published aggregates before trusting the reconstruction), pool the
2 x n_reps outcomes, permute the cell-A/cell-B labels n_perm = 20,000
times (matching exp12's convention), and report the two-sided
permutation p-value for |Delta_A - Delta_B|.

Seeds: M1 as specified above (100000/200000 for the three-way
comparison; 300000 for the fresh FAR check). M2 uses NO new random
draws for either matched point (both are reconstructions of published
grids through their original seed bases) except the permutation
labels themselves, seeded at 20260725 (distinct from exp12's
2026-07-20-derived draw, avoiding any accidental correlation between
the two permutation studies).

## 2026-07-25 — R4 PRE-REGISTERED: GARCH mechanism (M1), composite paired SE (M4), combined windowed statistic (M5), before any of the three scripts is run — plus three items resolved from EXISTING data, no new run

Six reviewer-requested items registered together; three require no new
simulation at all, found by locating existing outputs rather than
assumed absent -- reported here rather than silently rerun.

**No new run needed (M2, M3, Question 3) -- table-number mismatch
flagged.** The request's "Table 5" and "Table 7" do not match this
draft's actual numbering (verified: PAPER_DRAFT.md's Table 5 is the
PELT localization table, Table 7 is INDPRO's FAR-target sensitivity;
the content described -- the phi=0.95-vs-0.99 Delta note and the
Kalman-vs-ARIMA composite gaps -- is actually Table 4 (Sec 4) and Table
8 (Sec 5) respectively). Content matched by the cited numbers, not the
label, before proceeding:
  - **M2 (phi=0.95 vs phi=0.99 paired SE)**: `exp19_paired_se_grid_v8.csv`
    already contains both per-phi paired SEs for the qvar_x1.5 cell
    (phi=0.95: Delta=0.112, SE=0.0181; phi=0.99: Delta=0.074,
    SE=0.0175) -- and PAPER_DRAFT.md already reports the combined
    unpaired SE and the "~1.5 SE" conclusion from them (Sec 5, the
    Table 4 discussion). Unpaired, not paired, is the correct
    combination here for the same reason as R3 M2's Point 2: phi=0.95
    and phi=0.99 are different DGP parameterizations, so shared seed
    integers do not produce exchangeable pairs. No new experiment;
    the answer already exists and is already correctly stated.
  - **M3 (four-corner sidedness x parameter-knowledge)**: re-reading
    `experiments/exp10_cusum_ablation.py` -- it ALREADY computes all
    four corners in one run (two_sided_estimated, one_sided_estimated,
    two_sided_known, one_sided_known), not the two the request assumed
    existed. `paper_assets/exp10_cusum_ablation.csv` has all four:
    a=0.554 (SE 0.0223), b=0.636 (SE 0.0215), c=0.970 (SE 0.0076),
    d=0.990 (SE 0.0045). Only (b) one-sided/estimated=0.636 is missing
    from the current prose (Sec 4 only cites a, c, d). No new
    experiment; the fourth corner already exists in the committed CSV,
    just not yet in the narrative text.
  - **Question 3 (fresh-draw FAR for the plain ARIMA rung)**:
    `lsc.eval.runner.run` computes `empirical_far` via the standing
    far_check=300000 block for EVERY method in EVERY grid it runs, not
    just the ones a script explicitly highlights -- `arima_var_cusum`'s
    fresh-draw FAR is already in every grid_v4/v5/v9 `*_far_calibration.csv`.
    grid_v4_varbench_core: 4.4% / 6.0% / 4.2% at SNR 0.1/0.5/2.0 (5%
    target, phi=0.95). No new experiment; already computed as a
    byproduct of the standard runner, just not previously surfaced in
    the FAR-check narrative the way GARCH's was (exp24).

**New runs (M1, M4, M5), pre-registered before implementation:**
  - **exp32 (M1)**: does GARCH's fitted conditional-variance path
    sigma2_t track the true pre/post-break regime, or is it flat in the
    floor cells specifically? Spearman correlation + AUC (Mann-Whitney
    U, rank-based) between sigma2_t (or raw/ARIMA's z_t^2 baseline) and
    the true regime indicator, pooled across all post-training time
    points x n_reps=500 replicates, same 2x2x3 grid as exp15. NEW
    evaluation-only seed block (400000+) -- no calibration/threshold
    needed for this diagnostic (unlike M2 above, there is no downstream
    paired-decomposition requirement here, so a fresh block costs
    nothing). Falsifiable prediction: if GARCH's floor result is a
    generative mismatch (GARCH structurally can't see a permanent
    variance step), sigma2_t's correlation/AUC should be near-chance
    specifically in the four already-identified floor cells (q-channel
    all SNR, r-channel SNR 0.5/2.0, all x1.5) while raw/ARIMA's z_t^2
    baseline is clearly above chance there; if instead sigma2_t tracks
    the regime about as well as the baselines even in the floor cells,
    the CUSUM wrapper being underpowered is the better explanation.
  - **exp35 (M4)**: paired SE/z-statistic for Table 8's Kalman-vs-ARIMA
    composite gaps, exp19's methodology applied to composite_kalman
    (published in grid_v1/grid_v5, method=lsc_composite) vs
    composite_arima (published in exp20_composite_on_arima.csv) --
    both use the standing calibration=100000/evaluation=200000 blocks
    with the same arena, so they ARE scored on the same simulated
    paths and a true pairing applies (unlike M2's cross-phi
    comparison). Reconstruction verified against BOTH published
    aggregates (composite_kalman AND composite_arima) before trusting
    any paired SE, same discipline as exp19. Three cells: r x1.5/SNR0.1
    (largest gap, 0.818 vs 0.226), r x1.5/SNR2.0 (0.910 vs 0.632),
    q x3/SNR0.1 (0.438 vs 0.248, q-channel analog) -- covers the
    required largest gap plus two more, per the request's discretion
    clause.
  - **exp36 (M5)**: new `make_combined_windowed_detector`
    (lsc/eval/detectors.py, max of windowed_raw_cusum_score and
    windowed_raw_var_score in one score path, one calibrated
    threshold), tested on a mixed-channel two-event sequence in both
    orderings (level-then-variance, reusing exp04's own
    level_then_var breaks but with the new detector set since exp04
    never tested windowed_raw_var or the combined detector on it;
    variance-then-level, new) -- identical arena/protocol to exp04/exp27
    (spec-SNR 0.5, rearm_frac=0.5/refractory=20, match window=100,
    standing seed blocks). Reports recall_break1/recall_break2/
    precision/F1 for raw_var_cusum (reference), windowed_raw_cusum
    (mean-only), windowed_raw_var (variance-only), and the new combined
    detector, both orderings.

Seeds: exp32 evaluation-only 400000+ (justified above); exp35 and
exp36 reuse the standing calibration=100000/evaluation=200000/
far_check=300000 layout (both require draw-for-draw comparability with
already-published numbers, same reasoning as R3 M1's correction).
n_reps=500 throughout except where noted. Outcomes logged with numbers
when resolved.

## 2026-07-26 — R5 M1 PRE-REGISTERED: full r-channel phi-sweep, before grid_v9b is run

Registered before implementing `configs/grid_v9b_r_phi_lo.yaml` (verified:
no such file exists at commit time). Table 3b's phi=0.99 r-channel
extension (R2 M1) is a single point; this brings it to the same
phi in {0.5, 0.8, 0.95, 0.99} sweep Table 4/Fig. 2 already runs for the
level-shift case. phi=0.95 (grid_v4_varbench_core) and phi=0.99
(grid_v9_r_phi99, R2) are already published -- only phi in {0.5, 0.8}
are new (`configs/grid_v9b_r_phi_lo.yaml`, q values identical to
grid_v6_phisweep's phi=0.5/0.8 rows for draw-for-draw comparability).
`experiments/r_phi_sweep_analyze.py` assembles all four phi values into
one table and reports Spearman(amplification, raw-minus-arima
advantage) per break size, matching phiqbreak_analyze.py's convention
(Table 4's own assembler) applied to the r channel instead of the q
channel. Same protocol throughout: n_reps=500, T=500, far_target=0.05,
train_frac=0.25, standing calibration=100000/evaluation=200000 blocks.
Falsifiable question: does the r-channel's "prewhitening wins"
ordering hold uniformly across the full sweep, or does R2 M1's
phi=0.99 estimated-rung breakdown (ARIMA losing to raw at the subtle
break near the unit root) appear gradually as phi rises, or only at
the endpoint? Either answer is reportable; a gradual onset would
sharpen R2 M1's finding into a genuine phi-dependent boundary rather
than a phi=0.99-specific anomaly.

## 2026-07-26 — R5 M2 PRE-REGISTERED: GARCH oracle break-aware diagnostic, before exp37 is run (design revised after a real problem was found in the original spec, confirmed with the user before building)

The original request ("fit GARCH separately pre/post the break, CUSUM
the result, compare to exp15's plain-GARCH grid") was found to be
self-defeating before implementation: a correctly-refit model's post-
break residuals are z ~ N(0,1) by construction -- there is nothing left
for a CUSUM to detect once the model is told the truth. A model that
adapts, by definition, stops looking anomalous. Flagged to the user
with a proposed resolution; confirmed before building anything.

**Revised design**: explicit ORACLE / mechanism-diagnostic (same status
as exp10/exp26/exp30's known-parameter columns -- not a new entry in
exp15's calibrated-FAR table). New `garch_detector.oracle_two_regime_
residuals`: z_single (exp15/exp32's existing single-regime construction,
unchanged) paired with z_oracle (identical pre-break, refit on the TRUE
post-break segment Y[break_time:] from break_time on). `experiments/
exp37_garch_oracle_break_aware.py` reports, per exp15/exp32's 2x2x3
grid: post-break mean(z^2) under each construction (z_oracle near 1.0
CONFIRMS the self-defeat property directly, rather than leaving it
assumed) and whether each construction's max CUSUM score crosses
exp15's ALREADY-CALIBRATED threshold. The informative comparison is
NOT "does the oracle detect better" (it structurally cannot) -- it is
z_single's post-break z^2 elevation over 1.0, which IS the exact signal
the plain-GARCH CUSUM is weakly accumulating: a small elevation means
little signal was ever available to extract (a wrapper-agnostic
finding); a large elevation the calibrated CUSUM still fails to convert
into alarms sharpens "the wrapper is the bottleneck," since perfect
break-knowledge cannot add power beyond what z_single already carries --
it only consumes the same signal that produced z_single's departure in
the first place. This distinguishes "GARCH's underperformance is a
wrapper problem" from "it's partly a fit-quality problem too, masked by
online estimation" -- the user's reframing of the original request,
sharper than the original spec.

Seeds: evaluation-only 400000+, IDENTICAL to exp32's block (reproduces
exp32's z_single bit-for-bit on the same replicates -- a self-
consistency check -- and there is no downstream paired-decomposition
requirement that would demand a fresh block, unlike R3 M1's exp30).

## 2026-07-26 — R5 M3 PRE-REGISTERED and PARTIALLY CORRECTED: ALFRED vintage extension to GS10/UNRATE, GDP deferred

Sec 9's existing text already flags this exact extension as deliberately
deferred ("a materially larger undertaking... this project's own
history shows that rolling-window protocol extensions done quickly
have twice introduced real bugs") -- re-read before touching any code,
confirmed with the user, and scoped down accordingly: GDP is quarterly
(n_train=60/n_monitor=20 per real_data.py's SERIES config, a materially
different decision-month grid, not a parameter swap) and is NOT
attempted here, deferred as its own follow-up. GS10 and UNRATE are
monthly with IDENTICAL n_train=120/n_monitor=60 windows to INDPRO's
existing protocol -- mechanical extensions.

New `experiments/realtime_check_multi.py` (realtime_check.py itself
left untouched, same convention as real_data.py/m6_fred.py):
parameterized by series config (fred_id, transform, episodes) instead
of INDPRO's hardcoded values. VERIFIED against the published
`paper_assets/rd_realtime.csv` before trusting any new series: run
with series="indpro", reproduces every published alarm month/data
month/vintage exactly, cell for cell.

Episodes drawn from each series' own existing event list
(real_data.py's SERIES dict): unrate reuses INDPRO's gfc (2007-12) and
covid (2020-02) episodes (UNRATE's own event list IS NBER_PEAKS); gs10
uses its own three events (1979-10 Volcker, 2008-12 ZLB, 2022-03).

**Correction found during the run, not before it**: the gs10 "volcker"
(1979-10) episode fails outright -- direct ALFRED queries confirm GS10's
vintage history does not extend that far back (404 at vintage dates
1979/1990/1994/1996-06, first 200 response at 1997-01-15). This is a
genuine data-availability limit, not a bug in the generalization (the
INDPRO verification run had already ruled out a code bug). It also
corrects Sec 9's own existing "-checked 2026-07-23: ALFRED serves
vintage histories for all three series" note, which had confirmed
EXISTENCE of vintage data for GDPC1/GS10/UNRATE in general, not
coverage back to 1979 specifically for GS10 -- the earlier check's
scope was narrower than the sentence implied, caught here rather than
carried forward silently. gs10's EPISODES dict corrected to drop
"volcker", keeping "zlb" and "hike2022" (both well within confirmed
coverage). Outcomes (unrate, gs10) logged with numbers when the reruns
complete.

## 2026-07-26 — Two scope decisions made by the author (not silently picked), to be implemented in the Major-Weakness-1 restructuring pass, not as standalone edits

**Major Weakness 2 (DGP scope)**: author chose to scope the title/
abstract explicitly to scalar linear-Gaussian AR(1) state-space models
rather than build a second DGP class through the full grid. Rationale
recorded: the existing single-cell AR(2) check (R2 M2) already shows
the trichotomy surviving one departure from AR(1); a full second-DGP-
class grid (new generative model, new detector calibration against it,
a repeated 2x2x3+ grid) is a materially larger undertaking than a
revision-cycle addition. No new simulation from this decision --
title/abstract wording only.

**Minor Weakness 4 (GS10 placement)**: author chose to move GS10 to a
clearly separated exploratory subsection rather than drop it -- keeps
the real, already-disclosed content (Volcker/ZLB/2022 findings,
including this round's rd_realtime_gs10.csv) while being honest about
its weaker evidentiary basis (partly-author-selected events) than the
NBER/McConnell-Perez-Quiros-dated series. The corrected multiple-
testing family (currently 39) shrinks to exclude GS10's tests once this
lands. No new simulation from this decision -- Sec 9 restructuring only.

Both deferred to the Major Weakness 1 restructuring pass (Task tracked
separately) since both are structural/placement edits of the same kind
that pass already needs to make, not standalone changes.

## 2026-07-26 — R6 PRE-REGISTERED: raw_cusum FAR-precision check (M1, design corrected after a premise check), phi-peak joint test (M2), systematic paired SEs across Tables 3/3b/3c/4 (M3) -- before any of the three scripts is run

**M1 premise correction (found before writing any code, not after)**:
the request assumed raw_cusum's Table 2 threshold is "a single pooled
threshold" shared across SNRs. Checked directly:
`paper_assets/grid_v1_far_calibration.csv` has three DISTINCT raw_cusum
thresholds (27.49/103.19/213.89 at SNR 0.1/0.5/2.0) -- `lsc.eval.runner.
run` already calls `calibrate()` once per (arena, method), so raw_cusum
is calibrated separately at each SNR from that SNR's own null, and has
been since M5. There is no pooled-threshold confound to remove.

The real, well-posed version of the concern, given the architecture
already separates by SNR: is the empirical-FAR drift (4.0%/6.2%/8.2%
at SNR 0.1/0.5/2.0, Table 1) a finite-sample threshold-ESTIMATION
artifact that inflates the detection-rate advantage, rather than a
pooling bug? `experiments/exp38_raw_cusum_far_correction.py`:
recalibrates raw_cusum at each SNR with n_reps=5000 (calibration
seed0=100000, a strict superset of the original 500 draws -- same
draw sequence, not a different one), checks each threshold's
out-of-sample FAR on 2000 fresh far_check=300000 draws, and rescoves
detection rate at BOTH thresholds on the SAME n_reps=500 evaluation
draws (seed0=200000) Table 2 itself uses. lsc_kalman_cusum included at
n_reps=500 only (Table 1 already shows it calibrating close to 5% at
every SNR, no drift to investigate there).

**M2**: adds phi in {0.90, 0.97} to the existing subtle x1.5 phi-sweep
(now {0.5, 0.8, 0.90, 0.95, 0.97, 0.99}), same q=SNR*(1-phi^2)*r
convention as grid_v6/v9b, all three SNRs. Tests the peak-shape claim
JOINTLY (Delta(0.95) > both Delta(0.90) and Delta(0.99) simultaneously)
via a permutation test on the paired per-replicate raw-minus-kalman
differences at all three phi values, not just the pairwise 0.95-vs-0.99
gap already reported. Seeds: standard blocks, phi=0.90/0.97 cells are
genuinely new draws (no published cell exists at these phi values).

**M3**: systematic paired-SE reconstruction across every cell in Tables
3/3b/3c/4 where both compared rungs share a seed base (true for
essentially all of them per the paper's draw-for-draw convention).
Reuses exp19/33's exact methodology: rerun through the ORIGINAL
config/seed bases, verify the reconstructed aggregate matches the
published rate exactly before trusting the pairing, per-replicate
difference -> paired SE, reported alongside the independence-bound SE
already implicitly used. Single output CSV (table, row identifier,
Delta, paired SE, independence-bound SE) rather than inline per-table
edits, per the request's own preference given the volume. Falsifiable
per the request: some cells may NOT tighten under pairing (raw and the
compared rung could be negatively correlated per-replicate in
principle) -- reported as found, not assumed uniform.

## 2026-07-27 — R7 PRE-REGISTERED: smoothed-ARIMA composite (D), non-CUSUM GARCH alarm rule (E), paired SEs for Table 2b (F) -- before any of the three scripts is run

**D (smoothed ARIMA state-proxy for the composite comparison).** exp20
already showed the composite-on-ARIMA gap (0.818 vs 0.226 at the
flagship r x1.5/SNR0.1 cell) is attributable to 6/11 features acting on
ARIMA's one-step-ahead `fittedvalues` as the state analog -- a
disclosed judgment call, not a controlled substitution. This asks
whether a two-sided (fixed-interval-smoother) state estimate closes any
of that gap. Design: `ARIMAModel.filter(Y, compute_smoothed=True)`
(new) returns `smoother_results.smoothed_forecasts[0]` from the SAME
frozen training-prefix (order, params) fit `filter()` already uses --
statsmodels' built-in Kalman smoother applied post-hoc, so this is
explicitly NOT causal (it conditions on the whole series, both past and
future, the same oracle-status caveat already given to exp37's
break-aware GARCH refit and `known_*_var_cusum_score`) -- fed into the
existing unmodified 11-feature composite machinery in place of
`fittedvalues`. Grid: r-channel, x1.5 vol_mult only (the flagship cell
and its two SNR neighbors, not the full 12-cell cross -- this is a
targeted follow-up on one already-published gap, not a new benchmark),
SNR in {0.1, 0.5, 2.0}, n_reps=500, same seeds as exp20. Any outcome
(closes/partially closes/doesn't touch the gap) is reportable as
requested.

**E (non-CUSUM alarm rule on the existing GARCH fit).** exp32 showed
GARCH's own conditional-variance path sigma2_t tracks the true regime
(AUC 0.522-0.628) even at cells where its calibrated CUSUM-on-
standardized-residuals alarm sits at the FAR floor -- consistent with
"this specific wrapper is underpowered" rather than "GARCH structurally
can't represent the break." Design: an exceedance-indicator CUSUM --
the SAME construction as `tail_exceedance`/`tail_shortfall`
(lsc.diagnostics.features, the exp05b heavy-tail repair already used
elsewhere in the paper, Sec 8.3) -- applied to log(sigma2_t) directly
(an up-arm at its own training-prefix q=0.90 quantile, a down-arm at
its q=0.10 quantile, max of the two one-sided CUSUMs), in place of a
CUSUM on GARCH-standardized residuals. Same 2x2x3 grid as exp15/exp32
(channel r/q x vol_mult 1.5/3 x SNR 0.1/0.5/2.0), n_reps=500, same
calibrated 5% FAR, same seeds. Reported alongside the existing
garch_var_cusum rate at all 12 cells, with particular attention to
exp32's 5 floor cells.

**F (paired SEs for Table 2b known-vs-estimated gaps).** exp40 covered
Tables 3/3b/3c/4 (raw-vs-ARIMA) but not Table 2b (known-vs-estimated,
exp26's 12 cells), where some 0.01-0.02 gaps are called "within MC
noise" without the same paired-SE treatment. Design: identical
methodology to exp40 -- reconstruct both rungs (known_raw_var_cusum /
known_kalman_var_cusum from exp26, plus raw_var_cusum / arima_var_cusum
reconstructed through their original grid_v4/grid_v5 config and seed
bases) through the SAME evaluation draws, verify each reconstruction
matches its published aggregate exactly before trusting the pairing,
per-replicate difference -> paired SE, reported alongside the
independence-bound SE already implicitly used for the "within MC noise"
calls. Same 12 cells as Table 2b, same CSV shape as exp40 (config,
cell, Delta, paired SE, independence-bound SE, tightens_under_pairing).

## 2026-07-27 — DGP-scope reviewer conflict resolved by the author: hold the line

A new reviewer's Major Weakness 2 / Missing Experiments #1 asked for a
genuine multivariate or regime-switching DGP through the full grid, as
an alternative to the existing title/abstract scoping to scalar
linear-Gaussian AR(1) state-space models. This directly conflicts with
a previous reviewer's explicit endorsement of that scoping decision
("the more honest of the two possible fixes... I accept it"). Put to
the author directly (not decided silently): **hold the line** --
keep the AR(1) scoping, no new DGP class, no new simulation. Rationale:
the scoping was already litigated in an earlier round (2026-07-26 "Two
scope decisions made by the author") and independently endorsed by a
subsequent reviewer since; a genuine second-DGP-class undertaking
(new generative model, new detector calibration against it, a repeated
2x2x3+ grid) is a materially larger scope addition than a revision
cycle should absorb, and the paper already has a single-cell AR(2)
departure check (R2 M2) showing the trichotomy surviving one deviation
from AR(1). To be implemented as a short Discussion/limitations note
responding to this specific reviewer, not a structural change.

## 2026-07-27 — R7 RESOLVED: D, E, F all run at n_reps=500, folded into PAPER_DRAFT.md

**D**: at the flagship r×1.5/SNR0.1 cell, the smoothed-proxy composite
partially closes the one-step gap (0.226 -> 0.382, vs. Kalman's 0.818);
at SNR 0.5 and 2.0 the calibrated threshold and detection rate are
bit-identical to the one-step composite (thresholds 35.9996/33.2922,
matching exp20 to full precision) -- confirmed as a real mechanism, not
a bug: 5 of 11 composite features read `est.innovations` (the filter's
one-step forecast error, unchanged by `.smooth()`), only 6 read
`est.filtered` (which the smoother does change), and exp22's own
attribution data shows those 6 features drive only 7% of alarms at the
one cell where smoothing helped and are presumably even less binding at
higher SNR. Written up in Sec 5, directly after the exp22 diagnostic
paragraph it narrows.

**E**: exceedance-indicator CUSUM on log(sigma2_t) does NOT rescue
exp32's 5 floor cells (AUC 0.522-0.628) -- stays within 0.01-0.07 of
the 5% FAR target at all five (q SNR0.1/0.5/2.0: 0.058/0.052/0.120; r
SNR0.5/2.0: 0.050/0.088). Large real gains instead appear at the two
coarse x3/SNR2.0 cells, already well above the floor under the plain
wrapper (r: 0.548->0.688; q: 0.338->0.746) -- a genuine finding, but at
cells that were never the ones motivating the experiment. Folded into
the Sec 10 Discussion bullet that previously called this "untested."

**F**: all 12 Table 2b cells reconstructed and reproduced exactly. Of
the 6 negative known-minus-estimated gaps the existing prose called
"within MC noise," only 1 (raw r x1.5/SNR0.1, 1.4 paired SE) actually
is; the other 5 are 2.1-3.0 paired SEs from zero -- small in absolute
magnitude (|gap| <= 0.018, both compared rates >= 0.95) but not
attributable to noise alone. Corrected "within MC noise" to a precise
paired-SE characterization in Sec 4, and fixed an adjacent miscount
("Ten of 12 raw-rung cells" -> "Nine of 12", verified by direct
recount against Table 2b). Pairing tightens the SE in 17/24 rung-cell
combinations tested; the other 7 go the other way, reported as found.

**DGP-scope conflict**: put to the author directly (AskUserQuestion,
not decided silently) -- hold the line confirmed. Discussion bullet
added (see entry above).
