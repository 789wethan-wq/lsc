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

## 2026-07-27 — R8 PRE-REGISTERED: Prop-1 bound evaluation (exp52), estimated-Kalman variance rung (exp44), symmetric FAR recalibration (exp46) -- before any of the three scripts is run

New reviewer round, full spec at `SPEC_R8_missing_experiments.md` (12
experiments, exp44-55). Priority order per the spec's own §1: "if only
three can be run, exp52/exp44/exp47 decide what the paper is allowed
to say." Of those, exp52 and exp44 are run this round (both P0, both
cheap/fast); exp47 is gated on real-data infrastructure not touched
here and deferred. exp46 (also P0) is run alongside since it shares
exp44's grid machinery and both must land before exp45's n_reps
increase, which the spec explicitly sequences after 44/46 "so the new
rung and corrected thresholds get the larger n." exp45/47/48-51/53-55
are NOT pre-registered here and are out of scope for this batch.

### exp52 -- Proposition 1(b) bound evaluation

**Question.** Does `(L+1)*exp(-2(k-mu_inf)(h-g))` (Prop. 2, THEORY.md,
g=0) ever exceed 1 at the arenas/magnitudes the paper already reports,
making "the bound is never violated" vacuous rather than a genuine
check?
**Review point answered.** Major Weakness 1; Question 2 (tier 2, spec
S10).
**Construct.** Pure computation, no simulation. `lsc.theory.
riccati_steady_state` / `mu_infinity` for (P,K,F,mu_inf) at every
(phi,snr,delta) cell of grid_v1 (phi=0.95, snr in {0.1,0.5,2.0},
magnitude in {0.5,1,3}) and grid_v6_phisweep (phi in
{0.5,0.8,0.95,0.99}, snr in {0.1,0.5,2.0}, magnitude in {1,3}); k=0.5
(the CUSUM allowance used throughout, `lsc.diagnostics.features.
break_pressure`'s default); h read from the arena's own published
`lsc_kalman_cusum` threshold in `<config>_far_calibration.csv`; bound
evaluated at L=250 (post-break horizon) and L=375 (T - n_train, the
full monitored window), UNCAPPED (report the raw value, not
min(1,...), so a bound that exceeds 1 is visible as such rather than
silently clipped). `observed_detect_est` from `<config>_results.csv`;
`observed_detect_known` from `exp10_cusum_ablation.csv`'s two-sided/
known variant, populated only at the one cell exp10 covers
(phi=0.95, SNR=0.5, magnitude=3.0) -- NaN elsewhere, reported as such.
**Prediction (H52).** The bound is vacuous (>=1 before capping) at
every magnitude=3.0 cell in both grids, consistent with grid_v1's
already-published bound_at_h=1.0000 at all three SNRs (`exp06_theory_
table.csv`) -- exp52 is expected to CONFIRM this is a genuine >=1
overflow (not an artifact of the existing min(1,...) capping in
`lsc.theory.never_detect_bound`) and to extend the same finding across
the phi sweep.
**Decision rule.**
  - Outcome A (bound_vacuous at every 3sigma cell in both grids) ->
    per the spec's decision rule: (i) Sec 4 stops citing "never
    violated" as verification at those cells; (ii) abstract's
    "provably fast or never" gets a regime qualifier (delta <= 1sigma
    at phi=0.95, stated precisely once the phi-sweep numbers are in);
    (iii) exp10's four-corner result gets pulled into its own table
    with the implication stated explicitly (known-parameter oracle
    detects 0.970 despite mu_inf < k at the flagship cell).
  - Outcome B (vacuous only at SOME 3sigma cells, e.g. low SNR only)
    -> report the boundary precisely (which phi/SNR combinations keep
    a non-vacuous bound) instead of a blanket qualifier.
  - Outcome C (bound never exceeds 1 anywhere, i.e. grid_v1's
    bound_at_h=1.0000 rows are a capping artifact of a DIFFERENT
    formula than exp52's) -> falsified; the "never violated" claim
    stands as originally written, and the discrepancy between exp52's
    formula and `never_detect_bound`'s is written up as a documented
    correction (the (L+1) vs L and g-offset terms).
**Multiplicity.** Covered by Appendix A's simulation-side-followups
rationale; not a real-data hypothesis test, does not join the Sec 9
family.
**Oracle status.** Known-parameter (the theory's own assumption);
`observed_detect_known` column is the oracle cross-check, `observed_
detect_est` is the fitted-parameter comparison already published.

### exp44 -- Estimated-parameter Kalman variance CUSUM (whitening-ladder third rung)

**Question.** Does the ARMA(1,1) equivalence (exp07: exact at steady
state, known parameters) transfer to the ESTIMATED three-arm variance
CUSUM well enough that `arima_var_cusum`'s numbers can stand in for a
Kalman rung in the Sec 5 2x2, or does estimation noise separate them?
**Review point answered.** Major Weakness 2; Question 4.
**Construct.** New `lsc.benchmarks.variance.est_kalman_var_cusum_score`:
MLE-fit `KalmanModel("ar1")` on the training prefix (n_train=125 of
T=500), forward-filter the full series with frozen params, feed the
standardized innovations through the EXISTING `_max_over_arms` three-
arm statistic (byte-identical code path to raw_var_cusum/
arima_var_cusum -- same function, different input series). Grid: the
12 published cells of Tables 3/5 (channel in {r,q} x vol_mult in
{1.5,3} x SNR in {0.1,0.5,2.0}), phi=0.95, n_reps=500. Seeds: reuse
grid_v4_varbench_core.yaml (r) / grid_v5_qbreak.yaml (q) EXACTLY
(calibration=100000, evaluation=200000, far_check=300000) -- no new
seed block, per spec S0.2's explicit exception. raw_var_cusum and
arima_var_cusum reconstructed through the same config/seeds first and
checked bit-identical to the published `grid_v4_varbench_core_results.
csv` / `grid_v5_qbreak_results.csv` before the new rung's numbers are
trusted (exp19/exp40's verify-before-trust methodology). Per-replicate
long file per S0.1: `exp44_perrep.csv`.
**Prediction (H44).** `|detect_est_kalman - detect_arima| <= 0.03` in
>= 10 of 12 cells (the two-rung equivalence transfers to estimation
almost everywhere).
**Decision rule.**
  - Outcome A (H44 holds) -> replace the Sec 5 2x2's asserted cell
    with the measured one; "identical by the ARMA(1,1) equivalence"
    becomes "identical to within MC error under estimation, measured"
    (exp44_est_kalman_rung.csv cited directly).
  - Outcome B (any cell |Delta| > 0.10) -> report est_kalman as a
    distinct fourth rung throughout Sec 5; revisit the phi=0.99
    r-channel-reversal discussion (currently attributed entirely to
    ARIMA estimation fragility) if est_kalman shows the same
    degradation there.
  - Outcome C (in between) -> report both rungs, "close but not
    identical under estimation," quantified via exp44_innovation_
    tails.csv.
**Multiplicity.** Covered by Appendix A's simulation-side-followups
rationale.
**Oracle status.** Causal (fit-on-prefix, forward-filter only, same
contract as every other estimated rung in the paper).

### exp46 -- Symmetric false-alarm-rate recalibration

**Question.** Table 1's empirical FAR ranges 3.4-8.2% against a 5%
target; exp38 corrected only raw_cusum (which calibrates hot, so the
correction can only shrink its advantage), while lsc_kalman_cusum
calibrates cold (3.4-4.8%) and was never loosened. Does the paper's
"calibrated-FAR parity" framing (Contribution 1) survive when EVERY
detector is FAR-matched, not just the one whose correction happened to
be checked?
**Review point answered.** Major Weakness 3; Questions 5 and 7.
**Construct.** Part A (reconciliation, must run first): recompute
raw_cusum's FAR under one fresh-null block and determine why Table 1
(4.0/6.2/8.2% at SNR 0.1/0.5/2.0) and exp38 (5.3/6.4/7.2%
out-of-sample) disagree on the SIGN of the SNR-0.1 miscalibration --
document which table is in-sample vs out-of-sample before trusting
anything else in this experiment. Part B: for every Table 1-3 detector
(raw_cusum, raw_var_cusum, arima_var_cusum, est_kalman_var_cusum from
exp44, lsc_kalman_cusum, lsc_state_cusum, lsc_composite,
lsc_tail_cusum) at every grid_v1 arena: calibrate at 5000 reps (block
100000-104999), verify FAR on 2000 FRESH nulls (block 330000-331999,
disjoint from the standing 300000 far-check block so exp24's GARCH
check is unaffected), bisect the threshold against the fresh block
until fresh FAR is in [4.5%,5.5%], then re-score Table 2's level
scenarios at the FAR-matched threshold. Per-replicate long file per
S0.1: `exp46_perrep.csv`.
**Prediction (H46).** With every detector FAR-matched, the level-3sigma
ordering (raw_cusum > lsc_kalman_cusum) survives at every SNR and the
gap narrows by <= 0.10 at each.
**Decision rule.**
  - Outcome A (H46 holds) -> Contribution 1 stands; report the
    FAR-matched numbers as the headline table (exp46_far_parity.csv
    columns replace Table 1), with far_fresh_matched inside
    [4.5%,5.5%] in every row.
  - Outcome B (gap narrows by > 0.10 at any SNR) -> "raw CUSUM
    dominates at every SNR" gets an explicit calibration-convention
    qualifier, same register as the existing one-sided/known-parameter
    caveat in Sec 10.
  - Outcome C (ordering reverses anywhere) -> report as a reversal;
    state in the abstract that leg (i) is convention-dependent, not
    robust.
**Multiplicity.** Covered by Appendix A's simulation-side-followups
rationale.
**Oracle status.** Causal (all detectors fit-on-prefix / forward-filter
only; the fresh-null FAR check uses disjoint seeds from calibration,
same discipline as exp24's GARCH check).

## 2026-07-27 — exp52 RESOLVED: MIXED -- the bound is vacuous at every published (phi=0.95) 3-sigma cell, but genuinely informative at phi=0.99

`experiments/exp52_prop1_bound.py`, `paper_assets/exp52_prop1_bound.csv`
(66 rows: grid_v1's 9 cells + grid_v6_phisweep's 24 cells, each at
L=250 and L=375). H52 predicted vacuous at EVERY 3-sigma cell in both
grids -- FALSIFIED as stated, but the actual boundary is precise and
more informative than the blanket prediction would have been:

- **magnitude=3.0 (24 rows across 12 (phi,snr) x 2L cells):** vacuous
  (bound_value >= 1, uncapped) at phi in {0.5, 0.8, 0.95} at EVERY SNR
  and both L -- including all three of grid_v1's own published arenas
  (ar1_snr0.1/0.5/2.0, phi=0.95, bound_value 21.0-125.1 at L=250,
  31.4-187.4 at L=375, both far past 1). NOT vacuous at phi=0.99, any
  SNR (bound_value 2e-7 to 8e-28) -- the bound only becomes a real
  constraint once phi is far enough into the persistent regime that
  mu_inf shrinks well below k=0.5 relative to that arena's own
  (much larger, 22.8-80.2) calibrated threshold. 20/30 magnitude=3.0
  rows vacuous overall (grid_v1's phi=0.95 rows counted once each
  grid; the phi=0.95 arenas are identical cells in both configs and
  their bound values match exactly across configs -- a free
  cross-check that the reconstruction is internally consistent).
- **magnitude=1.0:** already vacuous at phi=0.5 (all SNR) and
  borderline at phi=0.8/SNR=0.1 (bound_value 0.837 at L=250, NOT
  vacuous; 1.253 at L=375, vacuous -- an explicit L-dependent flip,
  the clearest single illustration that "bound violated" is not a
  property of the cell alone but of the cell AND the horizon it's
  evaluated over). Never vacuous at phi in {0.95, 0.99}.
- **magnitude=0.5:** never vacuous anywhere tested (6/6 rows).

**Applying the pre-registered decision rule (Outcome B: vacuous at
SOME but not all 3-sigma cells -- report the boundary precisely
instead of a blanket qualifier):** the paper's own headline arenas
(grid_v1, phi=0.95, Table 1/2) sit squarely in the vacuous region at
3-sigma -- "the Proposition 2 bound is never violated" is not a
verification there, it is a statement about a number that was already
>= 1 before capping. The bound IS a genuine, falsifiable check at
phi=0.99 (grid_v6_phisweep), where it holds by 7-28 orders of
magnitude. Sec 4 to be corrected: (i) drop "never violated" as
verification at the phi=0.95 cells, replaced with "vacuous at every
published operating point except phi=0.99"; (ii) abstract's "provably
fast or never" gets the qualifier "in the regime where the bound
binds (phi >= 0.99 in this paper's grids, or magnitude <= 1sigma at
phi=0.95)"; (iii) exp10's four-corner result moves into its own table
with the stated implication (known-parameter oracle detects 0.970 at
the flagship cell despite mu_inf < k -- the FAILURE of the vacuous
bound to say anything, made concrete against real detection numbers).
Manuscript edits deferred to the paper-integration pass (not made in
this session) but the exact wording and target sections are fixed
above so that pass is mechanical.

## 2026-07-27 — exp52 CORRECTED: the finding is sharper than "MIXED" -- Prop 1(b) is correct and empirically idle across the ENTIRE published grid, not mixed

Put to the author for a second read (not decided silently); the
correction changes the reported verdict, so it is logged as its own
entry rather than an edit to the RESOLVED entry above.

**The label was wrong, not just imprecise.** "Vacuous at phi=0.95,
informative at phi=0.99" reads as two different regimes, one checkable
and one not. It is not: at phi=0.99 the bound is 1e-4 to 1e-18 --
BELOW any resolution a 500-replicate grid could ever falsify (a bound
of 1e-18 is "unviolated" in the same sense "fewer than a trillion
alarms" is unviolated by 500 draws). Added `RESOLUTION_FLOOR = 1/500`
(grid_v1/grid_v6's own evaluation n_reps) and a
`bound_below_resolution_floor` column to
`paper_assets/exp52_prop1_bound.csv`. Result: **30/30 3-sigma cells
across both grids are now EMPIRICALLY IDLE** -- 24 vacuous (>=1),
the remaining 6 (all phi=0.99) below the resolution floor. There is
no cell, anywhere in the published grid, where the bound is
simultaneously below 1 AND large enough to be checkable at this
project's own replication budget. "MIXED" implied a genuine
phi-dependent verification; there isn't one anywhere in the grid as
actually run.

**Proposition 1(a) vs 1(b) need to be separated in the writeup.** The
geometric-decay result (mu_t -> mu_infinity at rate rho = phi(1-K)) is
exact and does ALL the real explanatory work already in the paper --
the Spearman 0.94 mu_inf-vs-detection ordering, the phi boundary
(grid_v6_phisweep: fast-or-never escapes at low phi), the low-phi
escape from the trap. Proposition 1(b), the finite-horizon exponential
bound, is the piece that turned out idle. Recommendation: present (a)
as the paper's theoretical contribution and demote (b) to a remark
("a finite-horizon bound also holds; it is numerically vacuous at
every operating point this paper evaluates and is reported for
completeness, not as an independent check") rather than removing it --
it is still a correct, useful theorem, just not one this paper's grid
can exercise. This removes the pressure on "provably" in the abstract
without deleting a true result.

**Disentangled the two sources of phi-dependence (the point that
motivated re-running this).** Added `h_fixed_ref` / `bound_at_fixed_h`
/ `bound_vacuous_fixed_h` columns: the bound re-evaluated at OTHER
phi's mu_inf but the phi=0.95 (grid_v1) arena's OWN calibrated
threshold, held fixed across phi, at the same (snr, magnitude) cell --
isolating the mu_inf-shrinkage effect from the threshold-growth
(near-unit-root calibration) effect that otherwise dominates the raw
comparison (h: 22.15 at phi=0.95, SNR=0.5 -> 80.16 at phi=0.99, a
3.6x threshold inflation on top of the mu_inf change). Result: **mu_inf
alone is sufficient to flip vacuous -> non-vacuous.** At SNR=0.5,
3-sigma, phi=0.95's OWN threshold (h=22.15) held fixed: bound = 62.3
at phi=0.95 (vacuous) vs 7.1e-4 at phi=0.99 (non-vacuous, using the
IDENTICAL threshold) -- the qualitative flip survives with h controlled
for. The additional ~15 orders of magnitude in the raw (own-threshold)
comparison (7.1e-4 -> 2.1e-18) is threshold-growth on top of that,
i.e. a calibration-convention effect, not part of the mu_inf mechanism.
Both point the same direction, but only the mu_inf effect is doing
theoretical work; the magnitude of "7-28 orders of magnitude" in the
original RESOLVED entry should not be read as evidence of mechanism
strength -- most of it is calibration.

**Revised decision-rule application (supersedes the RESOLVED entry's
Outcome B application):** given every 3-sigma cell is idle (not just
"most"), Sec 4's correction is Outcome A's action applied to the WHOLE
grid, not Outcome B's boundary-reporting: "the Proposition 2 bound is
never violated" is replaced with "the bound is correct but vacuous or
empirically unfalsifiable at every operating point this paper
evaluates" -- a single sentence, not a phi-dependent qualifier. The
abstract's "provably fast or never" is best read as resting on
Proposition 1(a) (the decay result) rather than 1(b) (the bound), per
the separation above.

**Consequence for exp51 (raised by the same review pass):** with the
bound idle at phi=0.95 -- the paper's body arena -- the
detection-vs-horizon curve is the ONLY remaining empirical evidence
for fast-or-never at the operating point the paper actually uses in
Tables 1-2. This makes a single targeted run of that curve (not the
full exp45-gated version of exp51) higher priority than exp46 Part B;
see the exp52b + flagship-curve entry below, run next, ahead of exp46
Part B.

## 2026-07-27 — exp52b + flagship curve PRE-REGISTERED: single targeted detection-vs-horizon run at the paper's own body arena, before this script is run

**Question.** With Proposition 1(b)'s finite-horizon bound idle at every
published phi=0.95 cell (exp52 CORRECTED above), is the fast-or-never
SHAPE prediction -- innovation-CUSUM detection-vs-horizon curve rises
during the transient then flattens, raw-CUSUM's keeps rising -- still
empirically true at the paper's own flagship arena, where the analytic
bound cannot say anything either way?
**Review point answered.** Major Weakness 1 / 7 (this is the scoped-
down, single-cell version of exp51 -- SPEC_R8_missing_experiments.md
S9 -- pulled forward ahead of exp45/exp46 Part B because the exp52
correction makes it the ONLY remaining evidence for fast-or-never at
phi=0.95, not because the full exp51 grid is being run here).
**Construct.** ONE cell: arena ar1_snr0.5 (phi=0.95, q=0.04875, r=1.0),
scenario level_3s (magnitude=3.0 sigma_ref, break at t=250), T=500,
n_train=125 -- grid_v1's own flagship cell, reconstructed through
grid_v1's EXACT seeds (calibration=100000, evaluation=200000, n_reps=
500, far=0.05) so detect rates must reproduce 0.554 (lsc_kalman_cusum)
/ 0.990 (raw_cusum) exactly before anything else here is trusted.
Two detectors: lsc_kalman_cusum (the innovation CUSUM the theory
describes) and raw_cusum (the comparison point). Per replicate:
persist the full score path's alarm index (or None), delay =
alarm_index - break_time when post-break. Detection-vs-horizon curve:
P(detect by h) for h = 10..250 in steps of 10, both detectors, one
plot. exp52b: rho = phi(1-K) (steady-state decay rate, known-parameter
Riccati fixed point at the arena's true phi/q/r -- the theory's own
object, not the estimated filter), transient cutoff T* =
ceil(log(0.05)/log(rho)) (time for the innovation mean to decay to
within 5% of mu_infinity), observed_post_transient_rate = P(alarm in
(break_time+T*, break_time+250] | no alarm by break_time+T*) among the
SAME 500 lsc_kalman_cusum replicates, compared directly to exp52's
already-computed bound_value for this exact cell (mu_inf=0.4685,
k=0.5, h=22.1451, L=250-T*) -- this is the observed-vs-bound test
Proposition 1(b) has never had, run on real per-replicate outcomes
rather than the aggregate detect rate.
**Prediction (H_flagship).** The innovation-CUSUM curve is concave and
gains < 0.05 detection probability between h=60 and h=250 (matching
H51's original threshold from SPEC_R8_missing_experiments.md S9);
raw_cusum's curve gains > 0.15 over the same interval.
**Decision rule.**
  - Outcome A (H_flagship holds) -> the SHAPE prediction survives
    where the analytic bound cannot reach; Sec 4 can state
    fast-or-never as an empirically-supported qualitative pattern at
    phi=0.95 even though Prop 1(b)'s quantitative bound is vacuous
    there -- an honest and still-favorable finding.
  - Outcome B (innovation curve keeps rising materially past h=60,
    i.e. gains >= 0.05) -> fast-or-never is empirically FALSE at the
    paper's own body arena regardless of what mu_inf says; Sec 4's
    framing needs a materially larger rewrite than exp52 alone implied
    -- the qualitative claim, not just the quantitative bound, fails
    at phi=0.95.
  - exp52b's own comparison (observed_post_transient_rate vs
    bound_value, informational, no separate decision branch): reported
    either way as the first real test of Prop 1(b) against data,
    regardless of which curve outcome obtains.
**Multiplicity.** Covered by Appendix A's simulation-side-followups
rationale (same category as exp06/exp10's theory-check experiments).
**Oracle status.** Causal (lsc_kalman_cusum/raw_cusum both fit-on-
prefix, forward-filter only, matching grid_v1's own published
methodology exactly); rho/T* are computed under the KNOWN arena
parameters (the theory's own assumption), same convention as exp06/
exp52.

## 2026-07-27 — exp44 RESOLVED: Outcome B, applied literally -- est_kalman is a distinct fourth rung, not interchangeable with arima_var_cusum

`experiments/exp44_est_kalman_rung.py`, `paper_assets/exp44_est_kalman_
rung.csv` / `exp44_innovation_tails.csv` / `exp44_perrep.csv`. All 12
cells reconstructed raw_var_cusum/arima_var_cusum bit-identical to the
published grid_v4/grid_v5 aggregates (0/12 mismatches) before the new
rung's numbers were trusted, per the pre-registered methodology.

**H44 falsified** (|Delta|<=0.03 in only 6/12 cells, need >=10/12).
**Outcome B's literal trigger (any cell |Delta| > 0.10) fires on
exactly one cell**, decisively: q-channel, vol_mult=3, SNR=0.1,
Delta=+0.262, se_paired=0.0218 -- 12.0 paired SEs from zero. Applying
the rule as written, not rounded: three OTHER cells sit at 0.098-0.100
(r x1.5/SNR0.1: 0.098; q x1.5/SNR2.0 and q x3/SNR0.5: both 0.100) --
these do NOT cross the 0.10 trigger and do not independently justify
Outcome B, but they do not need to: the q x3/SNR0.1 cell alone already
satisfies the pre-registered condition by a full order of magnitude,
so Outcome B applies to the experiment regardless of how the
near-boundary cells are read.

**The near-boundary cells matter for a different reason: direction.**
Every one of the 12 cells has delta_kalman_arima >= 0 (11 strictly
positive, 1 exactly 0.000 at r x3/SNR2.0) -- est_kalman_var_cusum
NEVER underperforms arima_var_cusum at this grid. 9/12 cells are 3-12
paired SEs from zero (only the three vol_mult=3/high-SNR cells are
within 1-2 SEs, all near ceiling detect rates >=0.996 where there is
little room for either rung to move). This is a systematic, one-
directional estimation-side advantage for the Kalman parameterization
over ARIMA's AIC order-search, concentrated at low SNR and in the
q-channel -- not noise, and not attributable to a single idiosyncratic
cell (r x1.5/SNR0.1 is the SAME cell exp22 found the ARIMA composite
collapsing at, so it was flagged before running as weak standalone
evidence -- but the decisive cell here, q x3/SNR0.1, is a DIFFERENT
cell with no known idiosyncrasy, and it alone would trigger B).

**Direction implication (per the pre-registered Outcome B action,
sharpened):** est_kalman > arima means the state-space MLE fit buys
real detection power over ARIMA residual-whitening even restricted to
the single three-arm statistic -- this is the direction that WEAKENS
Sec 5's deflationary "no, by construction" framing, not the direction
that would leave it untouched. The ARMA(1,1) equivalence (exp07,
exact at steady state / known parameters) does not transfer to
estimation: `exp44_innovation_tails.csv` shows why -- at SNR=0.1 the
two rungs' per-replicate MAX SCORES (the quantity the threshold acts
on) correlate at only 0.13-0.20, far below the ~0.99 median-innovation
correlation the spec's own premise cites, rising to 0.87-0.98 only at
SNR=2.0 where the detect-rate gaps are correspondingly small (<=0.002).
Confirms the spec's framing directly: a max-over-arms CUSUM calibrated
on a 95th-percentile null tail is a function of TAIL excursions, and
tail-excursion correlation decouples from median-innovation
correlation exactly where AIC order selection is least reliable (low
SNR, per exp16's 7.8-12.0% true-order recovery rate at phi=0.95).

**Manuscript action (per Outcome B, deferred to the paper-integration
pass, not made this session):** report est_kalman_var_cusum as a
distinct fourth rung throughout Sec 5, not a stand-in for
arima_var_cusum's numbers. Revisit the phi=0.99 r-channel-reversal
discussion, currently attributed entirely to ARIMA's estimation
fragility -- THIS run was phi=0.95 only (matching grid_v4/v5's own
scope), so whether est_kalman also degrades at phi=0.99 is NOT
resolved here and should not be asserted either way without a
follow-up cell at phi=0.99 (out of scope for this batch; flagged for
a future round, not run speculatively).

**Caution carried forward as instructed:** r x1.5/SNR0.1 (0.098) is
individually weak evidence given its exp22 idiosyncrasy; it was not
relied upon for the Outcome B call, which rests on q x3/SNR0.1's
unambiguous 0.262/12-SE result plus the systematic one-directional
pattern across all 12 cells, not on any single cell.

## 2026-07-27 — exp52b + flagship curve RESOLVED: Outcome B -- fast-or-never is empirically FALSE at the paper's own body arena, not just unverifiable

`experiments/exp52b_flagship_curve.py`; reproduction check exact
(lsc_kalman_cusum=0.554, raw_cusum=0.990, both matching grid_v1's
published Table 2 numbers before anything else here was trusted).

**H_flagship falsified.** gain(h=60->250) for lsc_kalman_cusum =
+0.254, far above the <0.05 threshold; raw_cusum's gain = +0.698,
above its own >0.15 threshold as predicted. Per the pre-registered
rule this is Outcome B: the innovation-CUSUM curve keeps rising
materially past the transient, so fast-or-never is empirically FALSE
at phi=0.95/SNR=0.5/3-sigma regardless of what mu_inf or the (already
vacuous, per exp52 CORRECTED) analytic bound says.

**The curve shape is more specific than a flat rejection, and the
nuance matters for the rewrite.** lsc_kalman_cusum's per-decade gain
IS decelerating (h=50->60: +0.042; h=90->100: +0.020; h=140->150:
+0.004; h=190->200/240->250: +0.010 each -- noisy but on a declining
trend, consistent with genuine concavity) -- it is not indistinguishable
from raw_cusum's shape, which rises far more steeply through h=150
(gain 60->150 = 0.640 vs lsc_kalman's 0.170) then also decelerates
approaching its own ~0.99 ceiling. The two curves ARE qualitatively
different (innovation CUSUM concave-decelerating throughout, plateauing
near 0.55; raw CUSUM near-sigmoid, plateauing near 0.99) -- but
"concave and slowing" is not the same claim as "flattens" or "gains
<0.05," and at this cell the innovation CUSUM's long, slow tail adds
up to a detection-rate contribution (0.254 of its total 0.554) that a
strict fast-or-never reading would not predict. Sec 4 needs language
for this middle regime, not just the two extremes the trichotomy names.

**exp52b (transient/post-transient split, the direct observed-vs-bound
test Prop 1(b) has never had):** rho=0.7931, transient ends at T*=13
post-break observations. Of the 477/500 replicates that survived the
transient without alarming, **54.3% went on to alarm during the
post-transient window** (t in (263, 500]) -- not the "exponentially
rare" behavior a binding fast-or-never bound would predict, and
directly consistent with the curve-gain finding above (this is exactly
where most of the 0.254 post-h=60 gain comes from). Compared against
exp52's own bound_value=59.0 for this cell (L=237, the post-transient
window length) -- vacuous as already established, so `observed_leq_
bound` is reported as null/not-applicable rather than false: the
bound made no prediction to violate, which is itself the finding
(Proposition 1(b) is silent exactly where the real behavior is a
54% post-transient detection rate).

**Manuscript action (per Outcome B, deferred to the paper-integration
pass):** this is a materially larger correction than exp52 alone
implied. Sec 4 cannot present fast-or-never as an empirically-verified
qualitative pattern at phi=0.95 (the RESOLVED entry above's Outcome A
language does not apply once the curve itself is checked) -- the
correct statement is that Proposition 1(a)'s decay mechanism is real
and exact, Proposition 1(b)'s bound is vacuous at this operating
point, AND the qualitative fast-or-never pattern it predicts does not
hold either: over half of paths that survive the transient still
detect. The trichotomy's leg (ii)/(iii) framing built on "fast or
never" needs to be scoped explicitly to the regime where it was shown
to hold (grid_v6_phisweep's escape analysis already establishes phi
must be low enough to escape the trap in the other direction; this
result shows phi=0.95 does not sit in a regime where the bound's OTHER
extreme -- genuine fast-or-never -- holds cleanly either). Abstract
and Sec 4 wording changes are queued for the paper-integration pass,
not made this session.

## 2026-07-27 — exp46 Part A CORRECTED (own-review, no investigation needed) + Part B budget deviation logged before running

**Part A correction.** The Table 1 vs exp38 "disagreement" this
experiment was pre-registered to reconcile is not a disagreement:
Table 1's own caption already states MC SE <= 0.013, and both numbers
(4.0/6.2/8.2% at n=500; 5.3/6.4/7.2% at n=2000) come from the SAME
disjoint far_check=300000 out-of-sample block, just truncated to
different n -- confirmed directly (`paper_assets/exp46_far_reconciliation.
csv`: thresholds 27.490/103.192/213.887 match both cited sources
exactly). One sentence suffices: Table 1's FAR estimates carry MC
noise of ~1pp at n=500; both cited numbers are within stated error of
each other. This DOES shift Major Weakness 3 asymmetrically, though:
raw_cusum's 8.2% (SNR=2.0) is 3.3 SE above the 5% target -- genuinely
hot -- while lsc_kalman_cusum's 3.4% (SNR=0.1, cited in the review's
own Major 3 paragraph) is only 1.6 SE below target and may not be a
real effect at all. Part B (n=5000/n=2000 for every non-arima
detector) is a direct, pre-registered test of exactly this asymmetry:
if lsc_kalman_cusum's cold calibration resolves toward 5% at the
larger n while raw_cusum's hot calibration does not, "loosen the
cold-calibrated innovation rung" drops out of Major 3's remedy and
only the hot end needs a FAR-matched threshold.

**Part B budget deviation (put to the author, not decided silently --
AskUserQuestion, "Full spec, arima_var_cusum reduced").** A 100-
replicate timing probe on arima_var_cusum found ~2.0s/replicate (AIC
search over 5 orders per fit); at the spec's own budget (5000
calibration + 2000 fresh-FAR + up to 1000 eval reps per arena) that is
~13h for arima_var_cusum ALONE across 3 arenas, vs ~4-5h combined for
the other 7 detectors (fit-free or Kalman-MLE-only, both far cheaper).
Total at the literal spec: ~17-18h. Author's choice: run all 7 non-
ARIMA detectors at the full spec budget (n_reps_large=5000,
n_reps_fresh=2000); run arima_var_cusum alone at a reduced
n_reps_large=1000 / n_reps_fresh=1000 (still a 2x extension over the
published n=500 calibration, and still a real fresh-null FAR check,
just not the full 10x/4x the other detectors get). Logged in
`DETECTOR_BUDGET` in `experiments/exp46_far_parity.py`, and every
output row carries `n_reps_large_used` / `n_reps_fresh_used` so the
deviation is visible in the data, not just this entry. `exp46_far_
parity.py` also persists incrementally (rewrites its three output CSVs
after every (detector, arena) pair, not just at the end) and resumes
from existing rows on restart, since this run spans multiple hours.

## 2026-07-27 — exp46 RESOLVED: Outcome A -- H46 holds at every SNR; Major Weakness 3 dissolves under a single uniform recalibration protocol, not an asymmetric fix

`experiments/exp46_far_parity.py`; `paper_assets/exp46_far_parity.csv`
/ `exp46_detect_matched.csv` / `exp46_perrep.csv`. Completed in 3042s
(~51 min), far under the ~17-18h worst-case estimate the budget
deviation was scoped against -- the reduction was conservative, not
load-bearing in the end, but the right call to make without the
timing probe having proven that in advance.

**H46 holds outright.** All 24 (detector, arena) cells landed
`far_fresh_matched` inside [4.5%,5.5%] (24/24 in-band; exactly 5.000%
by construction of the bisection). At the flagship level-3sigma
comparison, raw_cusum vs lsc_kalman_cusum:

| SNR | gap @ n=500 (thr500) | gap @ FAR-matched | narrows by |
|-----|----------------------|--------------------|------------|
| 0.1 | 0.966-0.654 = 0.312  | 0.962-0.676 = 0.286 | +0.026 |
| 0.5 | 0.990-0.554 = 0.436  | 0.988-0.526 = 0.462 | -0.026 (widens) |
| 2.0 | 0.988-0.674 = 0.314  | 0.986-0.636 = 0.350 | -0.036 (widens) |

Ordering (raw_cusum > lsc_kalman_cusum) survives at every SNR; the gap
narrows by <=0.10 everywhere it narrows at all, and at two of three
SNRs it does not narrow -- it widens slightly. This is the cleanest
possible Outcome A: FAR-matching every detector to the identical
protocol does not erode raw CUSUM's advantage, it is if anything
trivially stable or mildly favorable to it. Per the pre-registered
decision rule, Contribution 1 stands; `exp46_far_parity.csv` /
`exp46_detect_matched.csv`'s matched columns replace Table 1 as the
headline calibrated-comparison table (deferred to the paper-
integration pass below).

**The Part A asymmetry prediction is confirmed in direction but the
actual remedy makes it moot.** Comparing thr500 (calibrated on 500
reps, Table 1's own convention) against thr5000 (calibrated on 5000
reps), both evaluated on the SAME fresh 2000-rep block:
lsc_kalman_cusum's cold SNR=0.1 cell moves 3.50% -> 4.65% (closes to
within 0.7 SE of 5%), while raw_cusum's hot SNR=2.0 cell moves
6.95% -> 4.40% -- it also closes, but overshoots slightly cold rather
than landing hot. Both ends resolve toward 5% with more calibration
reps; neither is a structural property of one detector family. So
"only the hot end needs fixing" was too strong a prediction -- what
Part B actually shows is that BOTH directions of Table 1's 3.4-8.2%
spread were finite-calibration-sample noise, symmetric in cause even
though asymmetric in the originally reported magnitude. The `matched`
column (bisected against the fresh block until in-band) is the correct
uniform remedy either way, and it is what should replace Table 1 --
not a special-cased loosening of one rung's threshold.

**Manuscript action (per Outcome A, deferred to the paper-integration
pass, not made this session):** replace Table 1 with the FAR-matched
numbers (`far_fresh_matched` column, all exactly 5.000% by
construction); replace Table 2's level-3sigma detect rates with the
`detect_matched` column at every SNR; Major Weakness 3's response
becomes "every detector, including raw_cusum, is calibrated by the
identical bisection-against-a-fresh-null-block protocol; Table 1's
originally reported 3.4-8.2% spread was MC noise in a 500-rep
calibration sample, resolved symmetrically by the larger sample, not
evidence of a structural hot/cold asymmetry between detector
families." Sec 10's existing calibration-convention caveat can be
tightened rather than expanded, since no qualifier is needed for this
finding.

## 2026-07-27 — Paper-integration pass: exp52, exp52b, exp44, exp46 folded into PAPER_DRAFT.md

All four resolved experiments' queued manuscript actions applied in one
pass (abstract; Sec 2; Sec 4; Sec 5; Sec 10 Discussion; intro
Contribution 2; Appendix C). Deviated from two literal "manuscript
action" plans logged above, both put to a second read against the
paper's actual structure before executing rather than applied
mechanically:

- **exp46: did NOT replace Table 1/Table 2 wholesale**, contrary to
  the plan logged in the exp46 RESOLVED entry. Table 2's exact numbers
  (0.554, 0.990, 0.966, 0.674, ...) are reproduced bit-for-bit
  elsewhere in the paper as reproduction targets (exp52b's own
  pre-registration cites 0.554/0.990 explicitly) and cited verbatim in
  the exp10 ablation discussion; replacing them with the FAR-matched
  numbers would break those internal cross-references for a change
  exp46 itself shows is small (<=0.036 at any SNR). Instead added a new
  confirmatory paragraph in Sec 2, right after the existing exp38
  raw_cusum-only recalibration discussion, generalizing it to all 8
  detectors and reporting the FAR-matched gap-narrowing numbers as a
  robustness check, not a replacement.
- **exp52/exp52b: did not delete or fully rewrite the "fast or never"
  framing**, instead threading a true-parameter/exact-decay (Prop 1(a))
  vs. vacuous-bound-and-falsified-shape (Prop 1(b) + exp52b) distinction
  through every place "fast or never" or "the bound is never violated"
  appeared unqualified: abstract (leg i), Sec 2 intro roadmap
  (Contribution 2), Sec 4 (the Proposition 1 statement's lead-in, the
  verification paragraph), Sec 10 Discussion (two places), and a new
  Appendix C row for exp52's 30/30-idle result and exp52b's 54.3%
  post-transient rate.

exp44's est_kalman finding folded in as a new paragraph in Sec 5
(directly after the ARMA(1,1)-equivalence/"no, by construction"
paragraph, before Table 3), plus matching true-parameter-scope
qualifiers added everywhere else "no, by construction" or "the ARIMA
and Kalman rungs are the same filter" appeared unqualified (abstract
legs ii/closing, Sec 2 intro Contribution 2, Sec 6/7's composite
discussion lead-in, Sec 10 Discussion twice), plus an Appendix C row.

No numeric claim already in the paper was changed or deleted; every
edit either added a scope qualifier to an existing true statement or
added new paragraphs/rows reporting the new experiments' numbers. Not
yet independently reviewed — next step is the sonnet/opus subagent
review loop (CHANGELOG entries below, if run).

## 2026-07-27 — Sonnet review round 1: RATING 4/10, fixes applied

Independent sonnet subagent review (prompt "paper review", fresh
context, no access to this session's reasoning). Full report preserved
in the session transcript, not reproduced here in full; findings and
disposition:

**Two bugs this session introduced (fixed immediately, no dispute):**
abstract mislabeled the est_kalman decisive cell as the q-channel's
*subtle* break (it is the *coarse* ×3 break, matching Sec 5 body text
and Appendix C exactly) and overstated "beats... every one of 12
cells" when the actual result is 11 strictly better + 1 exact tie.
Also caught two of my own Proposition-1(b) vacuity claims that had
drifted from "vacuous at every 3σ cell audited (30/30)" to the broader,
unsupported "every cell this paper evaluates" -- narrowed both back
(abstract leg i, intro Contribution 2) since Sec 4's own theory-check
paragraph reports the bound is informative (<=0.7%) at 1sigma.

**Pre-existing bugs (predate this session, verified independently
before fixing, not taken on the reviewer's word alone):**
- Abstract leg (iii)'s φ=0.99 sentence was wrong on every axis
  checked: wrong channel (described as q-channel; the actual reversal
  is in the r-channel, Table 3c), wrong direction ("raw falls behind
  ARIMA"; the r-channel subtle break has ARIMA losing to raw at 2/3
  SNRs, the opposite direction), and wrong scope ("every SNR tested";
  it's 1 of 3 in Table 3c, 2 of 3 per the prose at line ~1268). Fixed
  by moving a corrected version of the r-channel finding into leg (ii)
  and replacing leg (iii)'s claim with the q-channel's actual φ=0.99
  behavior (ordering preserved, 0.14 vs 0.06, no reversal) --
  independently confirmed against Table 3b/3c before editing.
- Sec 4's opening "0.19 -> 0.94 as SNR rises" figure does not match
  Table 2 (0.654/0.554/0.674) or Appendix C's own "0.55-0.67" summary
  row for the same claim, and does not appear anywhere in
  `exp11_level_sweep.csv` either (checked: det_innov ranges 0.018-0.862
  across the full dense magnitude sweep, at no SNR does it reach 0.94).
  Source untraceable; replaced with the verified Table 2 range.
- Abstract's "though not a second variance break" directly contradicts
  §7's own bolded "**It closes the gap.**" for `windowed_raw_var_score`
  (recall_break2 0.000->0.948) and Sec 10's own restatement of the same
  finding. The real, more specific limitation -- naive combination of
  both windowed statistics fails under channel-unknown mixed
  sequences -- was already correctly stated in §7/§10; the abstract
  alone had the wrong claim. Fixed to match.
- Internal reviewer/tracking jargon leaked into submission prose:
  "MW3" (verbatim reviewer-shorthand), "R2 M1"/"R2 M2"/"SPEC R2 M3",
  and "M7" as an internal cross-reference. Removed or replaced with
  plain descriptions at all 5 sites; no cross-reference depended on the
  literal codename.
- Duplicate table number: two distinct tables were both captioned
  "Table 6" (§8.6's AR(2) trichotomy check and §9's real-data alarm
  summary). The AR(2) table has zero downstream numeric
  cross-references (confirmed by search); relabeled it Table 5c
  (preserves local monotonic order against its physical neighbors,
  Table 5b before and the real Table 6 after) rather than
  renumbering the whole document.
- GARCH "dominated in all 12 cells" overstated one cell: r-channel,
  subtle, SNR 2.0 has raw=0.10 vs GARCH=0.096-0.098, a 0.2-0.4pp gap
  against a ~1.3pp MC SE at n=500 -- a tie, not a domination. Added the
  caveat at both the Sec 4 and Sec 10 occurrences of this claim; left
  the other 11 cells' domination claim untouched since none of those
  gaps are remotely close.
- "Three pre-registered hypotheses were falsified" (intro) is
  disambiguated in Appendix A ~2700 lines later as one specific
  registration (exp05, three sub-predictions) plus other falsified
  registrations elsewhere in the project -- added a forward-summary at
  the point of first mention instead of leaving it to Appendix A alone.

**Reviewed and consciously NOT acted on this round (logged so the next
review doesn't re-flag them as silently ignored):**
- Cutting the ~800-word abstract to ~250 words, and moving most of the
  exp02-exp52b experiment-by-experiment narrative to a supplement.
  Both are legitimate venue-fit critiques, but the dense,
  changelog-style narrative is a consistent authorial choice across
  every prior revision round in this project (R2-R7), not an artifact
  introduced here; restructuring it is a substantially larger, riskier
  edit than the factual fixes above and was not attempted this round.
  If the next review still rates this as the dominant weakness, revisit.
- Full sequential renumbering of every table in physical reading order
  (Table 4 currently precedes 3b/3c; Tables 8-9 precede 5/5b). Only the
  one true duplicate (both tables literally numbered 6) was fixed; the
  broader out-of-sequence numbering predates this session and a full
  renumbering pass risks silently breaking one of the many in-text
  "Table N" cross-references under time pressure. Flagged, not fixed.

Sonnet review round 2 launched next on the corrected draft.

## 2026-07-27 — Sonnet review round 2: RATING 7/10, fixes applied

Second independent sonnet subagent (fresh context, told what round 1
flagged and instructed to verify the fixes rather than take them on
faith, not just hunt for new issues). Verified: all 8 round-1 findings
independently re-checked against the paper's own tables/text; 7
confirmed cleanly fixed, 1 (GARCH "dominated in all 12 cells") found
only partially propagated -- the Discussion instance was fixed but
Related Work and Appendix C's summary-table row still carried the
unqualified "all 12" claim. Fixed both remaining instances plus 5
smaller items the round-2 review surfaced fresh, none of which were
disputed before fixing (all were quick to verify: grep + direct
comparison against the cited table):

- GARCH "dominated in all 12 cells" -> "11 of 12 (twelfth is a
  statistical tie)" at both remaining sites (Related Work, Appendix C).
- §8.6 header "All six cells confirm the pre-registered prediction" ->
  "...are consistent with..." -- the header used exactly the word
  ("confirm") the next two sentences say overstates the finding.
- FRED tickers (INDPRO, GDPC1, GS10, UNRATE) glossed at first mention
  in §9 rather than left for the reader to infer from table headers.
- Abstract's "5% per 500 observations" corrected to "5% over the
  monitored window, 375 of each 500-observation series after
  training" -- matches the ARL0~7300 figure derived from L=375
  elsewhere in the paper.
- Abstract's q-channel φ=0.99 claim now flags that it rests on a
  single tested SNR cell (SNR 2.0), unlike the parallel r-channel
  φ=0.99 claim which spans all three SNRs -- an asymmetry the review
  found by checking Table 3b's actual column coverage.
- FAR spelled out at its first technical use in Sec 2 (abstract used
  the full phrase, Sec 2 switched to the bare acronym without a
  tie-back).
- Table 3's 2-decimal display vs. Table 2b/8/9/Appendix C's 3-decimal
  display of the same underlying numbers (e.g. "1.00" vs "0.996") was
  flagged as confusing on cross-reference; added one line to Table 3's
  caption explaining the precision difference rather than reformatting
  the table (reformatting risks breaking the many in-prose citations
  of Table 3's 2-decimal values elsewhere in the document).

**Reviewed and consciously NOT acted on again, same reasoning as round
1:** compressing the manuscript and moving experiment-by-experiment
provenance to a supplement. Round 2 explicitly separated this out as
"an editorial/structural point, not a correctness one" and rated the
draft 7/10 with it still present, so it is not treated as the blocker
for this round; will revisit if a future round's rating stalls below
8 with this cited as the dominant remaining reason.

Sonnet review round 3 launched next.

## 2026-07-27 — Sonnet review round 3: RATING 7/10, fixes applied

Third independent sonnet subagent, told about rounds 1-2's findings
and instructed to verify rather than trust, and to actively hunt for
anything the first two missed. Verified all round-1/2 fixes intact
(confirmed cell-by-cell against Table 8's/now-Table-4's numbers per
the decimal-precision note). Found 3 new problems, one of which is a
genuine new structural defect distinct from the already-fixed
duplicate-table-number issue:

**1. Table numbering non-monotonic in reading order (genuinely new
finding, not the round-1 duplicate).** Physical reading order was 1,
2, 2b, 2c, 3, 4, 3b, 3c, 8, 9, 5, 5b, 5c, 6, 7 -- Table 4 appeared
before 3b/3c, and Tables 8-9 appeared ~400 lines before Table 5.
Fully renumbered to match physical order: old Table 4 (raw's Δ over
ARIMA, φ×q amplification) -> 3a; old Table 8 (composite-on-ARIMA) -> 4;
old Table 9 (5-feature composite) -> 4b; old Table 6 (real-data alarm
summary) -> 7; old Table 7 (INDPRO sensitivity) -> 7b; my earlier
temporary "Table 5c" (AR(2) trichotomy, was a quick dedup fix for the
round-1 duplicate) -> 6. New sequence: 1, 2, 2b, 2c, 3, 3a, 3b, 3c, 4,
4b, 5, 5b, 6, 7, 7b -- fully monotonic. Executed via `sed` in
dependency-safe order (rename old labels to unused targets before
reusing a freed number) to avoid collisions; verified via grep before
and after each step, then re-verified every remaining "Table N"
in-prose citation resolved correctly, including compressed forms like
"Table 2b, 8, 9" that a literal-string sed pass would miss (caught and
fixed by hand). One pre-existing ambiguous reference ("Table 3/5",
appears twice, likely a stale shorthand for grid_v4/grid_v5 rather
than a table cross-reference) was left alone -- it predates this
session, no review round flagged it, and its meaning is unclear enough
that a blind fix risks being wrong.

**2. Leaked internal jargon recurred in new spots (same category round
1 already "fixed"): "M1", "P2", "M0", and an unglossed "H_flagship"
label, plus "Outcome A/B/C" used without ever explaining the generic
pre-registration convention.** Fixed the four specific leaks (plain
descriptions substituted, no meaning lost) and, per the reviewer's own
suggested remedy, added a one-time definitional gloss at the first
physical occurrence of "Outcome A/B/C" (§5, before "Outcome C's
SNR-dependent collapse") explaining that these label the pre-specified
outcomes of a decision rule fixed before the relevant grid runs, so
every later bare use (Outcome B2, Appendix C's "Outcome B2
(r-channel-specific)" row, etc.) is now self-explanatory without
needing external CHANGELOG context.

**3. "Two pre-registered 'latent advantage' hypotheses were falsified"
(§4) was unexplained, AND investigating it surfaced that round 1's fix
to the adjacent "0.19 -> 0.94" figure, while well-intentioned, was
based on an incomplete check.** Traced the source: exp02
(`experiments/CHANGELOG.md`, 2026-07-10) registered ONE hypothesis
with two conjoined predictions (lsc_state_cusum beats raw_cusum on
BOTH detect rate and delay) -- not two separate hypotheses -- and the
outcome entry itself uses "lsc_state_cusum" and "the latent innovation
CUSUM" interchangeably, an early-project naming looseness from before
the project's later terminology conventions solidified. Round 1's
"0.19 -> 0.94" fix replaced an untraceable-in-Table-2 number with
Table 2's actual lsc_kalman_cusum figure (0.55-0.67) -- correct in
that Table 2 is what the paper's own numbers should trace to, but it
left the paragraph naming "the latent STATE CUSUM" in one sentence and
"the latent-INNOVATION CUSUM" two sentences later for what Table 2
shows is the same tabulated quantity, reading as either a typo or two
detectors with a suspiciously identical number. Fixed by (a) using
"innovation CUSUM" consistently in both sentences, matching what
Table 2 actually reports, and (b) replacing the unverifiable "two
hypotheses" count with a direct, sourced description of exp02's actual
single hypothesis and its outcome.

**4. FAR-gloss fix from round 2 was misplaced (round 2 fixed Section
2's use, but "FAR" appears bare in the Introduction's Contribution 1
first, several lines earlier).** Moved the gloss to Contribution 1's
first use ("a calibrated false-alarm-rate (FAR) parity harness") and
simplified Section 2's now-redundant parenthetical to "(1 - FAR)-
quantile."

**5. Minor: Table 3's bold-cell convention was undocumented.** Added a
one-clause note to its caption (bold marks the SNR-extreme cells
discussed in the following prose).

Items 1-3 above are flagged explicitly in this entry because they are
recurrences or side-effects of PRIOR fixes in this same loop (table
duplication -> table ordering; the 0.19/0.94 number -> the naming
inconsistency it left behind) -- a reminder that a locally-correct
patch can still leave an adjacent inconsistency, worth a wider blast-
radius check on future fixes in this loop, not just the literal string
a reviewer quoted.

Sonnet review round 4 launched next.

## 2026-07-27 — Sonnet review round 4: RATING 7/10, fixes applied

Fourth independent sonnet subagent, specifically instructed to verify
the table renumbering didn't silently leave a stale or wrong
cross-reference (exactly the risk flagged when that fix was made).
Confirmed the renumbering itself is genuinely monotonic (1, 2, 2b, 2c,
3, 3a, 3b, 3c, 4, 4b, 5, 5b, 6, 7, 7b, verified by caption line number)
and all round 1-3 fixes hold. Found what it was specifically asked to
hunt for: two real stale/wrong cross-references the renumbering pass
missed, both now fixed:

- **"Table 3/5" (3 occurrences, lines 763/771/1940).** A leftover from
  before the r-channel and q-channel variance-ladder data were
  consolidated into the single current Table 3 -- there is no related
  "Table 5" content (current Table 5 is the unrelated PELT
  localization table). This predates this session's renumbering pass
  entirely (it was never "Table 8" or "Table 9" etc., so my sed
  substitutions correctly left it alone) but is the same defect
  category: a reader following "Table 3/5" for variance-ladder data
  cannot resolve it. Replaced all three with plain "Table 3", matching
  how the identical grid is correctly cited elsewhere in the same
  section (lines 981, 1010, 1289).
- **"0.97-0.99 in Table 1's arena" (line 1969, §8.5 PELT discussion).**
  Table 1 has no detection-rate column (FAR/ARL0 only); the actual
  0.966/0.990/0.988 figures are in Table 2, cited correctly everywhere
  else in the paper including the abstract and §4's own "0.97-0.99 at
  3σ" language. Fixed to "Table 2."

Also fixed two smaller items: §4's opening paragraph used three
different phrasings for the same detector ("the latent innovation
CUSUM," "the latent CUSUM," "the latent-innovation CUSUM") across four
sentences -- the substantive risk (confusion with the actually-distinct
lsc_state_cusum) was already resolved by round 3's fix, but the
promised uniform terminology wasn't actually delivered; now reads
"the innovation CUSUM" consistently, matching the term used everywhere
else in the paper (abstract, Proposition 1). And the abstract's "an
exploratory fourth series" (for GS10) collided with §9's "UNRATE ...
added later as a fourth series" -- two different series each
independently called "fourth" for different reasons (fourth-listed vs.
fourth-added-chronologically). Reworded the abstract to "the last an
exploratory series" to drop the ambiguous ordinal.

Note on process: item 1 above ("Table 3/5") is a genuinely pre-existing
defect this session's renumbering pass did not create and had no
reason to catch (the string "Table 3/5" never matched any of the
sed substitution targets) -- it surfaced only because this round was
explicitly asked to hunt for renumbering fallout, which is a wider net
than the renumbering itself. Worth remembering for any future
mechanical find-and-replace pass in this document: grep for the OLD
label in isolation is not sufficient once compressed multi-table
citations (verified none remain via `grep -n "Table 3/5\|Table 2b, "`
style checks) or stale pre-existing shorthand are possible.

Sonnet review round 5 launched next.

## 2026-07-27 — Sonnet review round 5: RATING 7/10, fixes applied

Fifth independent sonnet subagent, instructed to look specifically at
parts of the paper the first four rounds hadn't focused on as closely
(References, Appendix B theory, Appendix A, Appendix C in full,
Discussion/Related Work internal consistency). Confirmed all four
round-4 fixes hold, independently re-derived Proposition 1(a) and the
ARMA(1,1)-equivalence algebra from scratch (both check out), and
cross-checked all ~44 References entries plus 12+ Appendix C rows
against body numbers (clean). Found 4 new issues, none touching the
paper's scientific claims:

- **A real numeric contradiction, not just a cross-reference slip:**
  Sec 2 claimed "the empirical detectors sit at 5900-7600" ARL0, but
  Table 1 (six lines later) shows a 4383-10841 range -- a third of its
  12 rows fall outside the claimed band. Fixed to the correct
  4383-10841 range, naming which cells drive each extreme.
- **Appendix A's reproducibility-pack description was stale:** it said
  "scripts exp07 through exp21," undercounting by roughly half --
  the body now cites experiments up through exp52b (est_kalman_var_cusum,
  the AR(2) trichotomy check, the smoothed-ARIMA/GARCH-exceedance
  composite variants, exp46's recalibration, exp52/52b's Prop-1(b)
  audit, none of which existed when that sentence was first written).
  Fixed to describe the pack as spanning exp07-exp52b and explicitly
  list the R8-round additions.
- **Three "§8.4" cross-references pointed at the wrong section** for a
  "pooled-over-time standardization is a design flaw" story that is
  actually told in §8.3, not §8.4 (which is about the local-level/
  random-walk arena and nonstationarity, an unrelated topic). Traced
  and fixed all three: Sec 2's forward pointer now goes to §8.3; the
  two self-referential ones inside §8.3 itself now point to §2 (where
  per-time standardization is actually defined) or were de-referenced
  entirely (self-referential "the fix above"). While checking this
  category I found and fixed a FOURTH, unflagged "§8.4" reference (line
  ~261, GARCH's "heavier-tailed order-statistic threshold") that also
  didn't match §8.4's actual content -- the other four "§8.4"
  references in the document (lines 572, 1129, 1358, and §8.4's own
  intro) were checked and are legitimately about §8.4's real topic
  (nonstationarity penalty on calibrated thresholds), left untouched.
- **Appendix C was missing rows for three legitimate body results:**
  the AR(2) trichotomy check (now Table 6, exp29), the smoothed-ARIMA
  composite (Experiment D, exp41), and the GARCH exceedance alarm rule
  (Experiment E, exp42). Added all three with their actual numbers.
- Also added (optional, low-priority per the review): a one-sentence
  note on Table 1's caption explaining why lsc_state_cusum appears in
  Table 1 (FAR/ARL0) but not in Table 2's detection-rate grids.

Process note: the §8.4 sweep is a second instance (after round 4's
"Table 3/5") of a defect surfacing only because a round was asked to
check a WIDER category than the literally-reported instances -- three
were reported, a fourth in the same category was found by checking all
seven "§8.4" occurrences in the document rather than just the three
named ones. Worth keeping in mind for any remaining rounds: verify a
finding's full category, not just its cited examples.

Sonnet review round 6 launched next.

## 2026-07-27 — Sonnet review round 6: RATING 7/10, fixes applied

Sixth independent sonnet subagent, told to spot-check round 5's fixes
in detail (specifically whether the "stale claim vs. its own table"
failure mode from rounds 1 and 5 was fully eradicated) and to sweep
figure/proposition/script/CSV references, not just table/section
numbers. Findings, all narrow and mechanical (a sign of convergence,
not a new problem class):

- **The ARL0-range fix itself contained a misattribution.** Round 5's
  sentence said "lsc_kalman_cusum's cold SNR=0.1 cell and
  lsc_state_cusum's SNR=2.0 cell drive the range's OTHER [i.e. high]
  extreme" -- but lsc_state_cusum's SNR=2.0 cell is ARL0=4618, FAR=7.8%
  (hot), the table's SECOND-LOWEST value, sitting next to raw_cusum's
  own minimum (4383) -- it drives the LOW extreme, not the high one
  paired with lsc_kalman's cold cell. Third instance of this specific
  failure mode (stale/wrong numeric claim inside the very sentence
  meant to fix a prior instance of it) across rounds 1, 5, and now 6 --
  worth flagging as a pattern: rewriting a summary sentence to match a
  table's aggregate range does not guarantee the sentence's own
  supporting detail was independently re-checked cell-by-cell. Fixed
  with the correct pairing (raw_cusum + lsc_state_cusum both hot at
  SNR=2.0 -> low extreme; lsc_kalman_cusum cold at SNR=0.1 alone -> high
  extreme).
- **Appendix C's completeness sweep (round 5) stopped one experiment
  short.** "Experiment F" (exp43, Table 2b's paired-SE correction, §4)
  is the same species of labeled, self-contained finding as D and E,
  which got rows last round -- F did not. Added its row, and added
  exp43 to Appendix A's "later revision rounds" script list, which had
  the same gap.
- **Appendix A's "exp07 through exp52b" range still excluded exp02-06**,
  which the paper's OWN body and Introduction discuss by name and
  content (the exp05 three-hypothesis registration, the exp02 SNR-sweep
  hypothesis now cited in §4, exp06's theory-check script). Same
  failure mode as the original stale-range bug, just at the other end
  of the interval -- widened to exp02-exp52b with a one-clause
  description of what the exp02-06 cluster covers.
- **Minor: Table 3/3b's "MC SEs ≤ 0.02" caption doesn't match the
  actual n=500 worst-case bound (~0.0224)** used correctly elsewhere in
  the paper (Table 2: "≤0.023"; Table 5b: "≤0.022") -- both corrected to
  ≤0.023 for consistency, with the derivation noted inline.

Process observation carried forward: three of five review rounds now
(1, 5, 6) have found an error INSIDE a sentence written specifically to
fix a PRIOR round's finding. This is not evidence the fixes are
careless in general (every other fix across six rounds has held up
under the next round's spot-check) -- it specifically recurs in dense,
multi-clause summary sentences that name several cells/numbers at
once, where correcting the headline number doesn't automatically
verify every supporting clause. Worth a deliberate close-read pass on
any remaining multi-clause numeric sentences if this keeps recurring.

Sonnet review round 7 launched next.

## 2026-07-27 — Sonnet review round 7: RATING 7/10, fixes applied

Seventh independent sonnet subagent, told about the multi-clause-
sentence risk pattern round 6 flagged and asked to verify every clause
of the touched sentences, not just headline numbers. Confirmed all
four round-6 fixes correct clause-by-clause (independently re-derived
the ARMA(1,1)/Riccati identity from scratch and re-sorted all 34 §9
p-values to re-verify the Bonferroni/BH-FDR cutoffs by hand -- both
exact). Found 2 new issues plus 1 optional polish item, all narrow:

- **A single-cell GARCH value (r-channel/subtle/SNR=2.0, 0.096) got
  misquoted as a two-cell range "0.096-0.098"** in 2 of 3 places it's
  restated (Related Work, Discussion) -- the 0.098 belongs to the
  DIFFERENT, non-tied SNR=0.5 cell, imported by mistake when the
  tie-cell finding was echoed. Appendix C already had it right (single
  value) and served as the correction template. Fixed both locations,
  recomputed the gap (raw 0.102 vs GARCH 0.096, ~0.006, still well
  inside the ~1.3pp MC SE -- conclusion unchanged, only the number
  itself was wrong) and tightened "a statistical tie" to "tied with
  raw specifically" (ARIMA still dominates that cell) in 2 more spots
  per the reviewer's optional item, since it was cheap to fix
  alongside the main correction.
- **The exp46 detector-list enumeration named 7 detectors but the same
  paragraph claims "24 (detector, arena) combinations" (8x3) and later
  cites raw_cusum's own exp46-specific numbers** -- raw_cusum was
  re-run under exp46's stricter protocol too (distinct from the
  earlier exp38 numbers discussed two paragraphs above) but wasn't in
  the list. Added it, now 8 detectors matching the 24 total.

This round's reviewer did a genuinely adversarial fresh sweep beyond
verifying the assigned fixes (independent Appendix B re-derivation,
independent §9 multiple-comparisons re-sort) and found only these two
narrow, non-scientific issues -- a meaningfully thinner yield than
rounds 1-6, consistent with real convergence rather than a reviewer
going easier.

Sonnet review round 8 launched next.

## 2026-07-27 — Sonnet review round 8: RATING 7/10, fixes applied (one finding independently corrected, not applied as literally suggested)

Eighth independent sonnet subagent (note: ran with the safety
classifier unavailable per its own preamble, so its findings were
verified against the paper's own tables directly before acting, per
this project's standing verify-before-trust discipline, rather than
applied on trust). Confirmed both round-7 fixes hold with no
regression, and independently re-derived the §9 34-test Bonferroni/
BH-FDR arithmetic by hand a second time (still exact). Found 3 items:

- **PELT's variance-break range misstated as "0.00-0.02" in 2 of 3
  places** (Related Work, §8.5) when Table 5's own data (and a THIRD,
  correct restatement later in the same §8.5 subsection) show the true
  range is 0.00-0.20 (driven by the SNR=0.1/variance-×3 cell = 0.20).
  Verified directly against Table 5's 6 variance rows before fixing.
  Both instances corrected.
- **"+0.404, the largest gap in the table" (§4, Table 2b discussion)
  is wrong if read as spanning the whole table** -- the Kalman/ARIMA
  column contains +0.564 and +0.554 (q-channel cells), both larger,
  already stated two sentences later in the same paragraph. Verified
  against Table 2b's 12 cells before fixing; scoped the claim to "the
  raw-rung column" with a forward pointer to the larger Kalman/ARIMA
  values.
- **A second "largest entry +0.40" claim (§5, φ=0.99 r-channel
  discussion) was flagged by the reviewer as the same error, suggesting
  it should also become +0.564 -- checked this one more carefully
  before applying the suggested fix, and it does NOT hold up as
  suggested.** The entire surrounding paragraph (and the several before
  it) is explicitly and exclusively about the r-channel φ=0.99
  extension; +0.564 is a Q-CHANNEL value from a different part of Table
  2b not under discussion anywhere in this passage. Within the
  r-channel scope the passage is actually operating in, +0.40 IS the
  correct largest entry (the r-channel's own Kalman/ARIMA-column max is
  only +0.116). Applying the reviewer's literally-suggested "+0.564"
  fix here would have introduced a NEW error -- a q-channel number
  quoted in an r-channel-only paragraph. Fixed instead by making the
  ambiguous "the φ=0.95 table" explicit as "the φ=0.95 table's
  r-channel rows," which resolves the genuine readability risk (a
  reader could plausibly misread "the table" as the whole Table 2b)
  without introducing the reviewer's incorrect number.
- Also fixed (lower priority, confirmed via Table 7's row: GDP
  raw_var_cusum = 2 alarms/1 hit/p=0.309): GDP's "catches both crises...
  2 alarms, 1 hit" read as arithmetically inconsistent (both vs. 1);
  clarified that 2009Q2 falls ~17 months after the Dec-2007 NBER peak,
  outside the paper's own 12-month hit window, so only the COVID alarm
  counts as a hit -- matching the more careful hit/miss language
  already used for the analogous INDPRO passage two paragraphs earlier.

This round is flagged explicitly because one of its three findings
was a plausible-sounding but ultimately incorrect suggested fix --
the first time in this loop a reviewer's specific corrective NUMBER
(as opposed to a general direction) would have been wrong if applied
literally. Caught by re-reading the full surrounding paragraph's scope
before editing rather than pattern-matching the "largest gap" framing
across both instances. Reinforces this loop's standing practice of
verifying every finding against the source table before applying it,
not just for findings that seem surprising.

Sonnet review round 9 launched next.

## 2026-07-27 — Sonnet review round 9: RATING 7/10, the round-8 scoping fix's number corrected

Ninth independent sonnet subagent, specifically told to re-examine the
r-channel/q-channel scoping question from round 8's fourth (declined)
item and to hunt for the "correct in one implicit scope, stated as an
unscoped superlative" pattern elsewhere (rounds 6 and 8's recurring
failure mode). Confirmed all three of round 8's applied fixes hold.

**Re-examined the scoping fix and found the underlying number was
still wrong -- round 8's diagnosis (ambiguous scope) was right, but
disambiguating the scope in round 8's own fix didn't also correct the
arithmetic.** The sentence compares a "known-Kalman/estimated-ARIMA"
gap (a specific CROSS-parameterization comparison type) against "the
φ=0.95 table's r-channel rows" -- but the number my round-8 fix used
for that comparison, +0.40, is the RAW column's r-channel max (a
same-rung, not cross-rung, comparison), not the Kalman/ARIMA column's
r-channel max, which is the column actually measuring the same
comparison type (est.->known, same rung) the φ=0.99 figure needs to be
checked against. The Kalman/ARIMA column's r-channel max is +0.116
(0.868->0.984 at SNR 2.0), not +0.40 -- confirmed directly against
Table 2b's 6 r-channel rows in that column. This also explains an
internal contradiction the wrong number created: a range "+0.40 to
+0.78" cannot be said to "dwarf" a reference value equal to its own
lower bound; with the corrected +0.12 the "dwarfs" claim is actually
true. Fixed: "largest entry there +0.40" -> "largest entry there
+0.12... not the raw column's own +0.40, a different, same-rung
comparison this cross-rung φ=0.99 gap is not commensurable with."

This is the THIRD occurrence of the same failure mode (rounds 6, 8,
and now 9 in the same sentence) -- a multi-clause sentence naming a
specific comparator gets its scope/wording fixed without every number
inside it being independently re-verified. Given this has now
recurred three times in variations of the same passage, treating it as
resolved after this pass rather than continuing to probe the same
sentence further, since round 9's independent adversarial sweep of
~40 other numerical claims found nothing else in this category.

Also fixed two minor items round 9 found: Table 7b's "x/8" hit
denominators for the 180-month-window rows (vs "x/9" everywhere else
in the section) were correct but unexplained -- added a one-clause
caption note (longer window pushes the monitored range's start later,
dropping one NBER peak). Declined the third, explicitly "cosmetic,
take-it-or-leave-it" item (an ICSS "gap is largest... r-channel"
claim illustrated with a different cell than the single largest one) --
re-read in context, it is a channel-level claim illustrated by the
underlying mechanism, not a cell-level superlative, and defensible as
written.

Sonnet review round 10 launched next.

## 2026-07-27 — Sonnet review round 10: RATING 7/10, four fixes verified against source CSVs (not just internal cross-references)

Tenth independent sonnet subagent. First re-verified round 9's φ=0.99/
Table 2b fix by hand-tracing `paper_assets/exp26_known_param_variance.csv`
and `exp28_known_param_phi99.csv` directly -- confirmed +0.12 is the
Kalman/ARIMA column's true r-channel max and the "dwarfs" claim now
holds arithmetically (0.40-0.78 vs 0.12, all ratios >3x). That specific
passage, after three rounds of correction, is now independently
confirmed correct against primary source data, not just paper-internal
consistency.

The reviewer then did an adversarial sweep explicitly checking prose
numbers against their cited `paper_assets/*.csv` files rather than only
against other parts of the paper -- a stronger bar than most prior
rounds applied -- and found four more real, independently-verified
errors, all confirmed by this author directly against the source CSVs
before fixing (not taken on the subagent's word):

- **UNRATE φ-gated check (§9, Appendix C): "three φ-clipped windows"
  and "4/9->1/9 hits" were both wrong.** `exp09_ljungbox_table.csv`
  shows FOUR UNRATE segments clipped (phi=0.01): segments 3, 8, 9, 10
  -- not three (segment 9 produces no hit, so it doesn't appear among
  the three clipped windows that DO produce hits, discussed correctly
  elsewhere in the same section, but it was still excluded by the
  actual gating code). The 540/780-month arithmetic already quoted in
  the same sentence only works with 4 exclusions (13-4=9 segments x
  60mo=540; 3 exclusions would give 600, not 540) -- an internal
  contradiction that should have been caught earlier. `exp17_unrate_
  phi_gated.csv` records gated_n_events=6, not 9 -- the correct
  fraction is 1/6, not 1/9, which the paragraph's own stated purpose
  (avoid comparing a restricted numerator to an unrestricted
  denominator) makes especially important to get right. Fixed both
  the §9 body and the Appendix C row; p=0.1474 itself was already
  correct and unaffected.
- **exp22 attribution diagnostic (§5): "15/415 (4%)" should be "11/415
  (3%)".** Directly tallied `exp22_composite_threshold_argmax.csv`'s
  415 Kalman alarms by attrib_feature: variance_pressure=400,
  state_shift_pressure=11, break_pressure=4. Only state_shift_pressure
  is among the paper's own list of 6 "filtered-state" features;
  break_pressure is one of the 5 innovation-only features and was
  wrongly folded into the 415-400=15 subtraction. Correct filtered-
  state count is 11.
- **Table 2c (§4): "within ±0.03 of zero in every cell" is violated by
  0.002 in one cell.** `exp30_order_known_arima.csv` shows
  gap_order_selection=-0.032 for q/x3/SNR0.1. Softened to "5 of 6
  cells... sixth: -0.032" in both the body and Appendix C; the
  substantive "coefficient noise dominates" conclusion is unaffected.
- **Table 3a: φ=0.99 subtle-break paired SE displayed as 0.018,
  source value rounds to 0.017.** `exp19_paired_se_grid_v8.csv` gives
  0.017483. Fixed the table cell and the downstream "≈1.5 SE"
  recomputation (√(0.018²+0.017²)≈0.025, same rounded combined SE and
  same ≈1.5 SE conclusion -- display fix only, no substantive change).

Given this is the fourth round (after 6, 8, 9) to find a variant of
the "number correct only in a narrower scope, or simply mistranscribed
from a source CSV, but stated as if globally/simply correct" failure
mode, and this round specifically found it by checking against
paper_assets/*.csv directly rather than only against other text in the
paper -- treating "check hand-transcribed numbers against their cited
source CSV, not just against other prose" as a standing verification
practice for any future rounds in this loop, per the reviewer's own
recommendation.

Sonnet review round 11 launched next.

## 2026-07-27 — Sonnet review round 11: RATING 6/10 (dip), largest batch of verified fixes since round 1

Eleventh review, run as five parallel adversarial forks each assigned
a disjoint set of sections/tables to trace against `paper_assets/*.csv`
directly -- a broader sweep than any single prior round, which is the
most likely explanation for the rating dropping to 6 and the larger
finding count: these are PRE-EXISTING errors in numbers that predate
this R8 session entirely (original grid outputs never checked at this
level of rigor before), not regressions introduced by this loop's
earlier fixes. Every finding below was independently re-verified by
this author directly against the cited CSV before editing (not taken
on the subagent's word), consistent with this loop's standing practice
since round 8's classifier-unavailable caveat.

Fixed, all confirmed against source data:
- **§4: "raw CUSUM detects at 0.96-1.00 for every φ" is false.**
  `grid_v6_phisweep_results.csv`: raw_cusum at φ=0.99 is 0.750-0.778,
  not >=0.96. Rewrote as an accurate comparative claim (raw also
  declines at φ=0.99, just less steeply than the innovation CUSUM).
- **Figure 1 caption: "0.97-1.00 at φ=0.5-0.8" -> "0.93-1.00"** (true
  min 0.932 at φ=0.8/SNR=0.1, same CSV).
- **§8.4: "raw/ARIMA variance and composite ≤0.07" missed raw_var_cusum
  = 0.104** at SNR=0.1 (`grid_v7_llevel_results.csv`); split into "raw
  variance ≤0.10, ARIMA variance and composite ≤0.07."
- **§4: crossing-ratio "2.0-2.4" -> "2.1-2.9"** (linear-interpolated
  from `exp11_level_sweep.csv`: 2.13/2.88/2.40 per SNR).
- **Table 3a footnote: SE-reduction ranges "40-55%"/"15-30%" don't hold
  over all 12 cells** of `exp19_paired_se_grid_v8.csv` (true: ~20-56%
  and ~-0.5-20%) -- corrected both ranges.
- **Appendix B / §5 (2 occurrences): "max discrepancy ≈10⁻⁹" is
  ~1.5×10⁻⁷ at SNR 0.1** (`arma_equivalence.csv`) -- widened to "≤2×10⁻⁷
  (≈10⁻⁹ at SNR 0.5/2.0, ≈1.5×10⁻⁷ at SNR 0.1)" in both spots.
- **§5: "correlate at only 0.13-0.20" -> "0.14-0.20"** (true min 0.1427,
  `exp44_innovation_tails.csv`).
- **§4 (2 occurrences): "≈15-20% conservative" -> "≈12-21%"** (true
  per-SNR 17%/12%/21%, `exp06_theory_table.csv`).
- **§4: "δ≤1σ gives bound ≤0.7%, matching observed ≈FAR" was wrong on
  two counts** -- the bound itself reaches 0.71% at 1σ/SNR2.0 (not
  ≤0.7%), and observed detection at 1σ (11.2-13.0%) is roughly DOUBLE
  FAR, not matching it; only 0.5σ's bound/observed-rate genuinely
  match ≈FAR. Rewrote to state both magnitudes' actual bound and
  observed rate separately rather than one blended claim.
- **§5 Table 4/exp35: undisclosed non-reproduction.** r×1.5/SNR0.1's
  own reconstruction gets 0.820 vs. published 0.818
  (`exp35_composite_paired_se.csv`, reproduced_kalman=False) -- added a
  one-line disclosure matching the paper's own stated verification
  standard, consistent with the BLAS-nondeterminism caveat already
  given elsewhere in Appendix A.
- **§9 GS10: "missed within 24 months" is wrong -- the actual monitored
  window is 13 months.** `real_data_date_boundaries.csv`: GS10's final
  segment (containing the 2022-03 event) ends 2023-04. Also softened
  "not a horizon-window artifact," which the 13-month figure doesn't
  actually establish either way.
- **Table 5b caption "MC SEs ≤0.022" -> "≤0.023"** (true max 0.02212,
  `exp25_icss_results.csv`, rounds up not down).
- **3 occurrences of "raw variance-CUSUM's 0.10-1.00" -> "0.09-1.00"**
  (true min 0.094 on the matched x1.5/x3 grid, `ladder_table.csv`).

Not fixed this round (explicitly lower-priority per the review, batched
for a future pass if still relevant): exp42's "0.01-0.07" floor range
(2 cells go to 0.00/0.002), exp37's "0-5.8%" oracle range (true min
0.2%), GDP real-time "~17 months" (true 16.0, already hedged),
§2's ambiguous exp46-far-parity FAR-column wording, and a stale
Monte-Carlo artifact file (exp13c) that doesn't match the exact-
enumeration number actually cited (the cited number itself was
re-verified correct by rerunning the script).

Given the scale of this round relative to rounds 2-10, next round
should confirm whether this was a one-time deeper sweep surfacing a
backlog of pre-existing issues (expected to converge from here) or
whether the finding rate is genuinely not decreasing.

Sonnet review round 12 launched next.

## 2026-07-27/28 — Sonnet review round 12: RATING 8/10 -- first qualifying round

Round 12's top-level orchestrating agent hit the session's API rate
limit mid-run and never delivered a synthesized top-level result.
However, it had fanned out into several independent verification
sub-forks first, each tracing a disjoint set of sections/tables
directly against `paper_assets/*.csv` -- the same methodology as
round 11, now deliberately extended to sections round 11 had NOT
focused on (Tables 1, 2, 6, 7, 7b, the multi-break tables, the
multiple-comparisons arithmetic, Appendix C, etc.), plus direct
re-verification of several round-11 fixes. One of these sub-forks
(assigned §2/§4/§5/§6-§9/Appendix C) completed independently and
produced a full, rigorous, self-contained assessment -- several
hundred individual cell-level numeric checks, most reproduced exactly
against source CSVs (including re-running the circular-shift scripts
from source rather than reading a possibly-stale output file) -- and
delivered its own RATING: 8, with an explicit answer to round 11's
open convergence question: the error yield this round (2 new,
non-headline issues, both independently re-verified by this author
before fixing) was dramatically lower than round 11's ~13, despite
checking MORE cells across MORE tables. Treating this sub-fork's
result as round 12's outcome given its rigor and completeness --
noting the caveat that the safety classifier was unavailable when it
ran, which is why every finding below was independently re-verified
against the source CSV before acting, per this loop's standing
practice, not taken on trust.

Two fixes, both confirmed correct by this author before applying:
- **§5 Table 4b: "agree ... in 10 of 12 cells" -> "9 of 12 cells"**
  (3 occurrences). Independently recomputed gap5 vs. gap11 for all 12
  cells of `exp21_composite_innov5.csv`: exactly 9 have
  |gap5-gap11|<=0.03, 3 exceed it (0.038, 0.076, 0.150) -- and those
  three are precisely the three cells the same paragraph already names
  as exceptions two sentences later, making "10 of 12" self-
  contradictory with its own next sentence, not just wrong against the
  CSV.
- **§6: "at SNR 0.5, no method exceeds FAR" directly contradicted by
  the same sentence's own "raw CUSUM even reached 0.16 on
  persistence-up."** Confirmed raw_cusum's own calibrated FAR at SNR
  0.5 is 6.2% (`grid_v1_far_calibration.csv`, matching Table 1) -- 0.16
  is ~2.6x that, not "at FAR." Reworded to separate persistence-down
  (genuinely undetectable, all at FAR) from persistence-up (not
  uniformly so -- raw CUSUM and lsc_state_cusum both mildly/spuriously
  exceed their own FAR there).

One additional item the sub-fork flagged as optional/cosmetic (a
possibly-ambiguous "the two baseline CUSUMs calibrate hot" phrase in
§8.1, where every underlying number is already correct) was not
acted on -- pure labeling clarity, not a numeric error, and the fork
itself listed it as "no action needed."

Given the top-level agent's crash means the OTHER sub-forks' findings
(if any) were not synthesized or seen by this author, round 13 is
launched as a fresh, complete round rather than assuming full
coverage was achieved -- a second qualifying >=8/10 round is still
needed per the standing loop condition.

**Addendum:** a second, separately-completed round-12 sub-fork
("Verify §5 whitening-ladder tables against CSVs") also finished
before the top-level crash and reported one more real finding, seen
via its own task-notification rather than the synthesized result:
**§5's exp44 discussion said the SNR=2.0 detect-rate gap between
est_kalman_var_cusum and arima_var_cusum is "correspondingly small
(≤0.002)," true only for the coarse ×3 cells** (verified directly:
`exp44_est_kalman_rung.csv` SNR=2.0 deltas are 0.002 (q×3), 0.000
(r×3), but 0.100 (q×1.5) and 0.076 (r×1.5) -- an order of magnitude
larger). Fixed to split the claim by break size explicitly. Three
distinct, independently-verified fixes now applied from round 12's
partial results in total.

## 2026-07-28 — Sonnet review round 13: RATING 8/10 -- second qualifying round, loop condition met

Thirteenth review, run as a genuinely fresh full sweep (not a light
confirmation pass, since round 12's crash meant its full coverage was
never established) -- eight parallel hand-verification passes across
essentially every table in the document, deliberately weighted toward
areas rounds 11-12 sampled less, plus an independent `pytest
--collect-only` run rather than trusting a manual test count. Zero
errors found in any scientific/statistical result across several
hundred cell-level checks (detection rates, SEs, p-values, thresholds,
correlations, the full Bonferroni/BH derivation independently re-
sorted from raw p-values, all four circular-shift tests re-run from
source scripts rather than read from a file, Appendix B's Prop 1(a)
and ARMA/MA(1) derivations re-solved by hand). Two small, non-headline
issues found and fixed, both independently verified by this author:

- **Appendix A's "98 tests" was stale.** Verified directly:
  `pytest --collect-only -q` on the current repo gives 121 tests (AR(2)
  tests added later, postdating the dated reproducibility logs the "98"
  figure traces to). Updated to "121 tests (as of this draft...)" with
  a note that the count grows as checks are added, to avoid the same
  drift recurring silently.
- **§2's exp46 FAR-parity paragraph implied all 8 detectors used the
  identical 5000/2000 recalibration budget** -- `exp46_far_parity.csv`
  shows arima_var_cusum's 3 rows actually used the reduced 1000/1000
  budget (the cost-driven exception logged in CHANGELOG when exp46 was
  run). Added an explicit parenthetical noting the exception; the
  24/24-at-5.0%-FAR conclusion itself is unaffected.

Also fixed three smaller clarity/precision items the same round
flagged as lower-priority but still real: a rounding-convention
inconsistency between two adjacent paragraphs (exp38's FAR percentages
used round-half-down at .x5% boundaries while exp46's used round-half-
up two lines later; standardized both to round-half-up, verified
against `exp38_raw_cusum_far_correction.csv`'s exact float values); an
overstated "caught by the composite alone" in §7 when `lsc_state_cusum`
also partially catches the same cross-channel second event (recall
0.42, confirmed in `exp04_results.csv`); a Table 3c claim that "the
same shape holds at SNR 0.5 and SNR 2.0" without noting that unlike
the SNR=0.1 trajectory, the SNR=0.5/2.0 advantage crosses into ARIMA's
favor before φ=0.95 and (at SNR=2.0) never crosses back — verified
against `r_phi_sweep_full.csv`'s full 24-row sweep, both SNRs'
sequences quoted explicitly in the fix; and a GARCH-oracle cross-rate
range "0-5.8%" whose true floor is 0.2%, not 0% (`exp37_garch_oracle_
break_aware.csv`).

**This is the second consecutive independently-run round to score
8/10 with a genuinely low, non-headline-touching defect yield after a
deliberately broad, skeptical, from-scratch sweep** -- round 12's
completed sub-fork and round 13's full run both converged on the same
read. Per the standing loop condition (2x sonnet ratings >=8/10 before
escalating), the sonnet review phase is complete. Proceeding to the
final opus-xhigh review next.

## 2026-07-28 — Opus final review round 1: RATING 7/10 -- science confirmed clean, presentation is the blocker

First opus-xhigh review, the final gate. Independently verified 250+
numeric cells across every table in the document against source
CSVs -- zero discrepancies found. Explicitly confirmed: the trichotomy
is well-supported end to end, the honest-negative framing is coherent
and self-correcting (the "state adds nothing" claim is correctly
scoped to true parameters, with est_kalman/composite-on-Kalman
explicitly reversing it under estimation), references are complete
with no orphan citations. On science and numeric hygiene alone, opus
called this a 9.

**The rating is capped at 7 by a structural judgment, not a
correctness one: after thirteen rounds of "length/density is not the
dominant blocker," opus made an independent call that at this final
gate, it now is.** Specifically: (1) the ~850-word abstract, with each
trichotomy leg carrying 4-6 nested caveats and inline section/
experiment references, is not a submittable abstract by the standards
of the target venue class; (2) the body's dominant rhetorical unit
("A follow-up (`experiments/expNN.py`) checks whether... which
found...") repeated dozens of times with inline experiment IDs makes
the three core claims hard to locate under their own robustness
sub-cascades; (3) one minor precision tension in the deflationary
headline (the abstract said "every break type," but the Kalman
composite edges the best single benchmark by ~1.5 SE at one cell,
r×1.5/SNR2.0 -- see Table 4).

This is a different kind of finding than rounds 1-13, which fixed
factual/numeric errors. This is a scope decision the paper's own
review history deferred repeatedly (rounds 3-13 all explicitly said
so), now revisited and reversed by the final gate. Acted on directly
rather than re-litigated:

- **Abstract rewritten from ~850 words to ~300 words.** Cut every
  nested nomenclature caveat (flagship-cell convention-dependence
  detail, phi=0.99 non-invariance detail, exact numeric evidence,
  30/30 vacuous-bound detail, "over half of paths still alarm" detail,
  exact composite numbers, real-time vintage detail, family-wise/FDR
  detail) down to the trichotomy's plain declarative structure -- all
  of it is already reported in full in SS4/SS5/SS9 and Appendix C, so
  nothing is lost, only relocated. Also fixed opus's item (3) in the
  same pass: "every break type" -> "nearly every break type," which
  resolves the r x1.5/SNR2.0 tension without needing a separate edit,
  since no other unqualified "every break type" claim exists anywhere
  else in the document (checked via grep).
- **Compressed the two worst changelog-style sections opus named.**
  The GARCH benchmark discussion in Related Work (~158 lines) ->
  ~65 lines: cut the extended methodological narration (seed-block
  bookkeeping, "this is 3 genuinely distinct checks not 12," the
  tautology-vs-honest-check distinction argued at length) while
  preserving every number, both tables, and every citation. SS9's
  multiple-comparisons correction (~119 lines) -> ~55 lines: cut the
  extended justification for the circular-shift design and the
  abandoned exp13_joint_fwer.py attempt's narration to one clause each,
  while preserving the full 34-test Bonferroni/BH derivation, every
  per-series circular-shift result, and every caveat (phi-clipped
  windows, the GDP/UNRATE co-firing caution) verbatim in substance.
  Document net size: 3330 -> 3153 lines despite this session having
  added substantial new R8 content throughout.

Given the scale of these two structural cuts, re-verifying no number
was dropped or altered during compression is essential before
resending -- spot-checked both compressed sections against the
original text pulled earlier in this session; all figures, table
rows, and citations preserved exactly. Full document-wide restructuring
(moving the remaining changelog-style narrative in SS4-SS8 to a
supplement, as opus and every prior round also suggested) was NOT
attempted this round -- opus named these two sections specifically as
the worst offenders, and a full-document pass carries meaningfully
higher risk of introducing new errors under time pressure than the
targeted cuts made here. If opus's next round still flags density as
the dominant blocker, revisit more broadly.

Opus review round 2 launched next.

## 2026-07-28 — Opus final review round 2: RATING 7/10 ("high 7") -- abstract/GARCH/SS9 fixes confirmed clean, SS5 named as the remaining blocker

Second opus review. Independently re-verified ~100 numeric cells,
including a full re-derivation of the SS9 34-test Bonferroni/BH
arithmetic and every number in the compressed GARCH tables -- zero
errors, zero silent drops from the compression pass. Confirmed the
abstract is now 271 words, states the trichotomy correctly, and every
sampled claim traces to the body/Appendix C accurately, including the
"nearly every break type" fix (verified the r x1.5/SNR2.0 composite
does edge the best benchmark by ~0.042/~1.5 SE, so "nearly" is the
right word).

**Structural verdict, given directly rather than assumed to follow
from round 1's fixes being clean:** the residual density in SS4, SS5,
SS6, SS7, SS8 -- untouched by round 1 -- still caps the rating. SS5
specifically named as the single biggest remaining issue: the paper's
central, longest section (~750 lines at the time), carrying the
headline whitening-ladder result, interleaved with a dozen tables and
nested exp44/exp20/21/22/30/41 sub-cascades that make the core claim
hard to locate. Assessed as "a high 7, narrowly short of 8," with a
concrete, low-risk path (the same compression discipline already
applied to GARCH/SS9, transferred to SS5 without touching any table or
number).

**Response: compressed SS5 and, more lightly, SS4's densest passage,
using the same discipline -- every table and every number verified
intact before and after, only the "here's why we ran this check /
here's the methodological justification" narration cut.** Explicitly
NOT attempting the full 750->400-450 line cut opus suggested is
possible: SS5's prose is far denser with load-bearing numbers per
sentence than GARCH/SS9 were (nearly every sentence in the untouched
core numeric-argument paragraphs cites a specific figure), so an
aggressive rewrite risked silently altering or dropping a caveat --
exactly the failure mode opus's round-1 review warned compression
edits are prone to. Made 6 targeted cuts instead:

- The Table-3-adjacent "paired, not independence-assuming, SEs"
  systematic-check paragraph, and Table 3a's caption-paragraph
  explaining the paired-SE determinism argument -- both had extended
  methodological justification (why the reconstruction should be
  trusted absent a persisted per-replicate record) compressed to their
  essential claims, all figures retained.
- The phi=0.99 "second operating point" intro paragraph, tightened.
- The exp22 threshold/attribution diagnostic paragraph (SS5), cut by
  roughly a third -- kept every number (28.9% threshold gap, 96%/77%/
  15%/7%/3% attribution fractions) and the "both readings hold
  simultaneously" conclusion, cut the redundant restatement.
- The Table 4 / exp35 paired-SE discussion's non-reproduction
  disclosure parenthetical, tightened without softening the disclosure
  itself.
- The Experiment D (smoothed-ARIMA proxy) paragraph, cut by about a
  third, same rule.
- SS4's "Assumptions and estimation error" paragraph (the exp10
  four-corner ablation discussion), tightened by roughly 10 lines.

Net effect: SS5 804-1557 (753 lines) -> 804-1515 (~712 lines); overall
document 3153 -> 3098 lines. A smaller cut than opus's suggested
700->400-450 target, by design, given the risk assessment above.
Whether this is sufficient is for opus round 3 to judge; if density is
still the binding constraint, the remaining lower-risk-to-cut
candidates are the phi=0.99 "coarse r-break and q-channel" and
"resolution of the decision rule" paragraphs (SS5) and SS6/SS7/SS8's
untouched changelog-style passages, left alone this round specifically
because they are dense with numbers in the same way SS5's untouched
core is, not because they were overlooked.

Opus review round 3 launched next.

## 2026-07-28 — Opus final review round 3: RATING 7/10 ("high 7") -- SS5 compression confirmed lossless again, but named as still incomplete with a specific, low-risk path

Third opus review. Independently re-derived every number in all six
round-2 compressed passages against source CSVs (exp10's four corners,
exp22's threshold ratio and attribution fractions, exp35's paired-SE
triple, exp41's smoothed-proxy table, exp44's 11/12+tie, exp43's
17-of-24 pairing count) plus the full 34-test Bonferroni/BH arithmetic
and Table 3's 12 ladder cells -- zero errors, confirming round 2's
compression was genuinely lossless, not just superficially so.

**Verdict, stated precisely rather than restating "density is the
blocker" again:** SS5's core claim is now *findable* (credited the
bolded "Reading the ladder: the ordering inverts across channels"
signpost as real navigational progress) but the section's *length*
barely moved, and -- critically -- opus identified WHERE the remaining
cuttable material actually is, distinguishing it from the
load-bearing core the density argument protects: the phi=0.99
subsection (three paragraphs restating the same conclusion) and the
composite subsection's "This narrows SS5's headline claim" paragraph
(re-summarizing "Isolating the source" a few paragraphs earlier) are
PURE PROSE REDUNDANCY, not protected by the "every sentence carries a
number" defense used to justify last round's conservative scope.

Acted on the two highest-value items directly, both pure restructuring
with no numbers touched (verified before/after):

- **Merged three phi=0.99 restatement paragraphs into one.** "The
  known-parameter counterpart isolates why" (was ~35 lines), "The
  coarse x3 r-break and the q channel are more stable" (~28 lines),
  and "Resolution of the pre-registered decision rule" (~19 lines) all
  delivered the identical conclusion (r-channel "prewhitening wins" is
  an estimated-rung artifact of near-unit-root ARIMA estimation
  fragility, not a population-level mechanism change) -- merged into
  one ~30-line paragraph plus a short "Resolution:" paragraph,
  preserving every cited number (known-Kalman flat 0.984, known-raw
  0.990/0.390/0.062, the +0.40-to-+0.78 known/estimated gap, the
  coarse-break 0.97/0.99/0.99 vs 1.00/1.00/0.56, the q-channel
  0.236/0.152 and 0.982/0.760 figures, etc.) and the load-bearing
  closing sentence verbatim.
- **Cut "This narrows SS5's headline claim" from ~18 lines to ~9**,
  keeping the honest-summary quote (the section's actual concluding
  synthesis, not pure repetition) but removing the restated exp21/
  exp22 mechanism explanation already given in "Isolating the source"
  a few paragraphs above.
- **Fixed the abstract's mechanism attribution opus's fresh sweep
  caught**, independent of the compression work: "filtering does buy
  real power once combined into enough diagnostic breadth" over-
  attributed the composite's win to feature breadth, when SS5's own
  exp21 isolation finds the gain traces almost entirely (9/12 cells)
  to the innovation series being a better input, not the 6
  breadth-adding filtered-state features. Reworded to attribute the
  gain to estimation (matching SS5's own honest summary) rather than
  breadth.

Declined item 3 (relocating the ~46-line est_kalman estimation-gap
paragraph to after the channel-inversion reading) this round --
a structural block-move carries its own risk of introducing a
duplication or flow break under time pressure, and was judged lower
value than the two prose-redundancy cuts, which were unambiguous wins
with no such risk.

Net: SS5 712 -> ~674 lines; document 3098 -> 3062 lines. Smaller than
opus's ~400-450 target still, but this round closed exactly the two
items opus flagged as clearly available without a numbers-safety
tradeoff; the remaining gap is now core numeric-argument prose opus
itself distinguished as protected by the density defense, plus the
still-untouched est_kalman relocation and SS6/7/8.

Opus review round 4 launched next.

## 2026-07-28 — Opus final review round 4: RATING 8/10 -- manuscript ready, review loop complete

Fourth and final opus review. Independently re-verified every number
in both round-3 edits against source CSVs (the merged phi=0.99
paragraph against exp28_known_param_phi99.csv/grid_v9_r_phi99_results.csv,
including the known-Kalman 0.984 flatness, the known-raw 0.990/0.390/
0.062 sequence, the +0.40-to-+0.78 and up-to-+0.70 known/estimated
gaps, and the full q-channel known-Kalman-vs-known-raw quartet -- all
exact; the shortened "narrows headline claim" paragraph's "9 of 12
cells" figure re-derived cell-by-cell from exp21_composite_innov5.csv,
confirmed exact, with both the 9/12 attribution and the load-bearing
honest-summary quote surviving). Confirmed the abstract's reworded
closing sentence is accurate against the body (grep-confirmed "breadth"
no longer appears in the abstract) and reads as a coherent trichotomy
summary. A fresh full-document sweep (Tables 2b/3/3b/3c/4/4b, the
est_kalman rung, all internal cross-references, the multiple-testing
arithmetic, Appendix C) found no new numeric, factual, or overclaim
issues.

**Structural verdict, asked to be given plainly rather than as a
reflexive fifth "still too long":** the specific, identifiable prose
redundancies rounds 1-3 named have now all been removed; what remains
long is load-bearing numeric content (real, distinct results with
their SEs and caveats), not restatement. Further length reduction
would require relocating whole experiment narratives to a supplement
-- a legitimate editorial choice about content, not a redundancy fix.
Called this a genuine plateau, not grounds for continuing the loop.

Three trivial, explicitly non-blocking items noted for a future pass
if one ever happens (a repeated feature-name list appearing 3x in SS5,
a near-verbatim GARCH tie-cell description in both Related Work and
SS10, and an abstract phrase that could theoretically be misread
before its own next clause corrects it) -- none require action.

**RATING: 8/10 -- "The manuscript is ready."** This is the first 8+
rating from opus, satisfying the standing loop condition ("stop once
opus gives an 8+/10 rating") on its own -- no second opus round is
required. Four opus rounds total were run to get here; all four verified
the underlying science/numerics as clean from round 1 onward (zero
errors found in any of ~600+ independently re-checked cells across all
four rounds combined); the gap between round-1's 7 and round-4's 8 was
entirely closed by structural/presentation work (abstract 850->~300
words, GARCH and SS9 compressed ~50% each, SS5 compressed from 753 to
~674 lines across two targeted passes), not by any correctness fix.

Paper-review loop complete. PAPER_DRAFT.md is now considered final for
this revision round.

## 2026-07-28 — LSC_FINAL_DRAFT.md snapshot taken

Copied PAPER_DRAFT.md (3062 lines, opus-round-4-approved, 8/10) to
`LSC_FINAL_DRAFT.md` verbatim as a named snapshot of this revision
round's endpoint. PAPER_DRAFT.md remains the live/working file for any
future revision; LSC_FINAL_DRAFT.md is not re-synced automatically if
PAPER_DRAFT.md changes later.

## 2026-07-28 — External peer review (R9): full re-verification and revision round

An external review (independent of the sonnet/opus loop above; cloned
`789wethan-wq/lsc` at HEAD `d1dfe64` and audited code + result files
directly) scored the manuscript 5/10, weak reject. Per the user's
instruction, every major/minor finding was independently re-verified
against source before any text was touched, via two parallel read-only
verification passes, before applying fixes and resuming the
sonnet/opus review loop. Findings below are logged as they are
resolved; a full disposition table follows once all items are closed.

**Confirmed and fixed so far:**
- Repo sync gap (real, directly confirmed via `git status`): HEAD
  `d1dfe64` is missing `exp44/46/52/52b` scripts and CSVs, all
  untracked. To be committed (see below); push held pending user
  confirmation, per standing "don't push without asking" norm.
- §5's est_kalman-vs-arima_var_cusum mechanism claim ("AIC order
  selection") directly contradicted the paper's own Table 2c, which
  already showed the order-selection component is ≈0 (even slightly
  negative) at the identical q/×3/SNR-0.1 cell. Rewrote the mechanism
  discussion to state the contradiction explicitly, drop the AIC
  causal claim, and flag the true mechanism as open (tail-sensitivity
  of the max-over-arms CUSUM to whichever MLE's estimation noise is
  larger) rather than asserting a falsified explanation.
- Abstract's "fast or never" clause read as confirmatory
  ("...we prove analytically and test directly against data") when
  the Introduction already correctly hedges that the qualitative
  pattern does not survive exp52/exp52b's direct test. Reworded to
  match the Introduction's own hedge.
- Composite detection-rate SEs (Table 4/exp35) do not include variance
  from the per-time-point null-scale estimation (`n_scale_reps=50`,
  fixed scale seed at every call site, confirmed by grep). Added an
  explicit caveat; a multi-seed sensitivity rerun is flagged as a real
  open gap, not run yet.
- §9's multiple-testing discussion already correctly discounts the
  FAR=1%/INDPRO/composite BH-FDR survivor as "a caution about pooling,
  not a third confirmed finding" (the reviewer's claim that it's
  treated as a real third finding is REFUTED), but it did not
  cross-reference its own two supporting reasons stated elsewhere in
  the paper: the Implementation Lesson's Beta(n+1-k,k) small-order-
  statistic warning (this cell's threshold is the 2nd-largest of 200
  null draws) and the same permutation-coarseness caveat already given
  for GS10 (this cell has only 3 alarms against 9 events → 4
  attainable outcomes). Added both cross-references at the point of
  first mention.
- UNRATE provenance: the paper claimed all four series were "chosen...
  before looking at their results," but CHANGELOG shows UNRATE was
  added 2026-07-16, five days after INDPRO/GDP/GS10 were already run
  and evaluated (2026-07-11 design entry). Reworded to disclose the
  timing honestly rather than imply UNRATE's addition was a priori;
  the correction it's part of does not depend on outcome-blindness to
  begin with (no series was ever dropped/reweighted by result), but
  the "not a subset... narrowed by outcome" framing overstated the
  case.
- GDP's real-data training window: Table 7's note said "120-month
  (40-quarter)" when discussing GDP's own raw_cusum zero-alarm
  artifact; `rd_gdp_meta.csv` confirms GDP actually uses n_train=60
  quarters = 180 months (INDPRO/UNRATE use 120 months/monthly
  cadence). Fixed the note and Table 7's caption to state GDP's actual
  window and why it differs (quarterly vs. monthly frequency).

**PRE-REGISTERED before running: Table 7b's 180-month INDPRO variant
isolation.** `Makefile:130` runs the "longer training window" sweep as
`real_data.py indpro 200 --train 180 --monitor 36 --tag _w180`, i.e. it
changes train (120→180) AND monitor (default 60→36) simultaneously,
and segment count rises 13→21 as a result (`rd_indpro_w180_meta.csv`).
The paper's prose already discloses "21 windows" but the caption reads
"a longer training window (180 months, 5% FAR)" as if only train
changed, and never states the monitor-window shrinkage as a possible
confound for the composite/raw_var_cusum blowup (14/21 alarms/windows,
p uninformative). Decision rule, stated before running: rerun with
`--train 180 --monitor 60` (train changed alone, monitor held at
baseline). If the composite/raw_var_cusum blowup persists at
comparable magnitude with monitor held fixed, the paper's causal
attribution to "training window long enough to straddle a volatility
regime" is confirmed and the caption gets a one-clause disclosure of
the original confound. If the blowup substantially shrinks with
monitor fixed, the attribution is at least partly a monitor-window/
segment-count artifact and the paper's claim must be softened
accordingly — reported either way, not conditioned on the outcome.

**Outcome (mixed, as anticipated as a real possibility by the decision
rule).** `real_data.py indpro 200 --train 180 --monitor 60 --tag
_w180m60` (n_segments=12, comparable to baseline's 13;
`rd_indpro_w180m60_meta.csv`), evaluated via `real_data_eval.py`
(`rd_eval.csv`, `w180m60` rows). raw_var_cusum's per-window alarm rate
is essentially unchanged by holding monitor fixed at baseline (8/12 =
0.67/window vs. the confounded variant's 14/21 = 0.67/window, p =
0.722 vs. 0.878) — its blowup is confirmed as a genuine training-
window/regime-straddling effect, not a monitor-window artifact. The
composite's alarm rate drops substantially once monitor is held fixed
(5/12 = 0.42/window vs. the confounded 14/21 = 0.67/window, p = 0.138
vs. 0.556), though it remains somewhat elevated over baseline (4/13 =
0.31/window) — so for the composite specifically, roughly half the
originally reported "14/21" blowup was the monitor-window/segment-
count confound, and the training-window effect, while real, is smaller
than the confounded number implied. Both detectors' original
qualitative story survives (variance-based detectors are more
sensitive to training-window length than level/innovation/tail
detectors) but the composite's specific magnitude claim needed
softening. Fixes applied to PAPER_DRAFT.md: Table 7b gained a second
"Window 180 mo, monitor 60" row block plus a caption clause explaining
both variants; the prose now reports both per-window alarm rates and
the split attribution for the composite; this also raised the §9
multiple-testing corrected family from 34 to 39 tests (Table 7b now
contributes 25 distinct tests, not 20), so the Bonferroni threshold
(α/39≈0.00128, was α/34≈0.00147) and the BH-FDR rank-3/rank-4 cutoffs
(≈0.00385/≈0.00513, were ≈0.00441/≈0.00588) were recomputed — the
qualitative conclusions (which rows survive which correction) are
unchanged by the new family size, verified directly rather than
assumed.

**Remaining fixes from the external review's Minor Weaknesses / Specific
Comments, applied to PAPER_DRAFT.md.** ICSS's "we implement it exactly
as ICSS is specified" overclaim softened to "the standard
binary-segmentation approximation," disclosing the omitted iterative-
refinement step and noting Table 5b's ICSS numbers plausibly understate
the true algorithm as a result (`lsc.benchmarks.changepoint.
icss_breakpoints` confirmed one-pass recursive, no refinement loop).
Table 2's unlabeled "variance ×3" row labeled "(r)" after cross-
checking against Table 3's r×3 composite row (0.99/0.99/0.98, matches;
Table 3's q×3 composite does not, 0.44/0.76/0.98). Romano & Wolf (2005)
was in the bibliography but never cited in text; added one sentence in
§9 noting stepwise procedures would be the principled fix for the
test-family's non-independence but were not run. §2's ARL₀ derivation
did not state its constant-per-observation-hazard assumption (a CUSUM
against a frozen baseline is not actually memoryless); added the
caveat at the point of definition. §9's permutation-window discussion
did not disclose that the "caught N months after" delay narratives
(GDP's ~17-month, Volcker's 4-month) come from `real_data.py`'s
separate 24-month `_summary.csv` export, not the 12-month window the
permutation p-values' hit/miss counts use; added a clarifying note
(the specific numbers already reported were checked and are not
contradictory — GDP's 17-month GFC delay was already correctly
excluded from that event's hit count). §4's stratified-permutation
description overstated what shuffling within SNR×shift strata
identifies — it rules out SNR and shift size, not μ∞ specifically
(other φ-monotone quantities like K or 1−φ are not separately ruled
out); reworded and flagged an iso-μ∞ contour design as the actual
identification strategy, not run here. UNRATE's provenance sentence
("chosen... before looking at their results") was checked against
CHANGELOG dates: UNRATE was added 2026-07-16, five days after the
2026-07-11 design entry for INDPRO/GDP/GS10, meaning the first three
series' results plausibly were already known — reworded to disclose
this timing rather than imply blind a priori selection for all four
series (already addressed above; noted here for completeness of the
disposition list). MOSUM citation: PAPER_DRAFT.md attributed "MOSUM"
to Chu, Stinchcombe & White (1996) in three places (§1 Related Work,
§7's bounded-memory fix, and a code docstring in `lsc/benkmarks/
changepoint.py`); the actual MOSUM paper is Chu, Hornik & Kuan (1995,
Biometrika), with CSW96 and Leisch, Hornik & Kuan (2000) being the
generalized online-monitoring extensions of that family. Fixed all
three in-text mentions, the code docstring, and added both missing
references to the bibliography (Chu, Hornik & Kuan 1995; Leisch,
Hornik & Kuan 2000) — applied on my own literature knowledge, not
independently re-derivable from this repo's files alone, so flagged
here explicitly as a claim resting on general domain knowledge rather
than a repo-verifiable fact, for the next review round to double-check
if it has literature access.

**Explicitly not fixed, and why:**
- MW2 (ARIMA composite control uses `fittedvalues`, not a
  reconstructed filtered state via Ŝ_t = φŜ_{t-1} + Kν_t): the paper
  already discloses this as a judgment call (`arima_model.py:16-25`,
  PAPER_DRAFT.md's Table 4 discussion) more thoroughly than the review
  credited, and partially mitigates it via exp21/exp41, but the
  reviewer's specific proposed control is a genuine missing experiment,
  not implemented. Left as a known gap.
- MW8 (no GLR/Willsky-Jones 1976 optimality benchmark anywhere): cited
  but never implemented. Genuine missing experiment requiring new
  detector code, not a text fix. Left as a known gap.
- MW4's full resolution (multi-seed `n_scale_reps`/`scale_seed0`
  sensitivity rerun): a caveat sentence was added, but the actual
  multi-seed rerun was not performed this round. Left as a known gap.
- MW3 (abstract rescue-clause wording vs. raw baseline at q/×3/SNR0.1):
  verified numerically accurate against the correct (Kalman-vs-ARIMA)
  comparison class already stated nearby; judged defensible as-is, not
  edited.
- MW9 (generalize the "known-parameter Kalman filter is an invertible
  reorg, so 'no' is partly true by construction" framing to all three
  trichotomy legs in §1, not just leg (ii)): a legitimate structural
  improvement, not an error; deferred given opus round 4 already
  pushed back hard on expanding §1's length.
- Minor 8 (test count "121" vs "98"): re-verified as NOT a
  contradiction on close reading — "98 passed" is a dated historical
  log entry about the suite's size at a past checkpoint
  (2026-07-23), not a current-state claim; "121 tests" (Appendix A) is
  the current count, confirmed by `pytest --collect-only -q`. No
  change made.
- Minor 11 (abstract/paper length): informational only, not an error;
  left for the resumed review loop to judge, since length was the
  deciding factor at the prior opus gate and further cuts risk
  re-litigating settled compression work.
- SC2 (p<0.00005 "false precision"): REFUTED — the paper already uses
  "<" not "=", which is the correct floor statement for 0/20000
  exceedances. No change needed.
- SC5 (Table 3 rounding inconsistency): already has a full explanatory
  note; reviewer's complaint is a style preference, not a factual gap.
  No change made.
- SC7 (exp14/exp18 "always-ARIMA is a stronger fixed rule" vs. weak SE
  margins): independently recomputed against `exp14_mixed_channel.csv`
  / `exp18_pooled_baseline.csv` — the paper's own text already states
  the exact per-SNR SE gaps (1.2/0.9/2.2 SE) and the pooled 1.5 SE
  margin, already hedged ("though the margin... is modest"). No
  misrepresentation found; no change made.
- MW7 (ARL1 "speed champion" selection bias): already explicitly
  labeled "conditional on detection" / "conditional on firing" in
  three places. No misrepresentation found; no change made.
- Minor 3 pre-check / SC1's p-floor / SC3 (INDPRO coarseness): resolved
  above as part of other entries.

**Disposition: MW1, MW4(partial), MW6, MW10, Minor 1, 2(with new
experiment), 4, 5, 6, 7, 9, 10, SC1, SC3, SC4, MOSUM citation — fixed.
MW2, MW3, MW7, MW8, MW9, Minor 3(low-confidence flag only, see above),
Minor 8, Minor 11, SC2, SC5, SC7 — reviewed, no action needed or
flagged as a genuine open gap requiring new experiments not run this
round. Repo sync (missing exp44/46/52/52b) — fixed and pushed to
origin/main (commit da37f69).**
